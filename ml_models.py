"""
This file handles classes necessary for Machine Learning models to predict Motility Induced Phase Separation (MIPS).
"""

### System Packages ###
import os
import gc
import pandas as pd
import numpy as np

### Sci-kit Learn Packages ###
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, KFold

### Tensorflow Packages ###
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential

### Torch Packages ###
import torch
import torch.nn.functional as F
import torch.nn as nn

### Graph Packages ###
import dgl
from dgl.nn.pytorch import GATConv
import networkx as nx

### XGBoost Packages ###
import xgboost as xgb

### Compute / Specific Packages ###
from ml_features import aggregate_features
from ml_utils import load_model
from TorchModels import TorchGAT
import gpu_utils


class PhasePredictModel:
    """ A Class for handling ML models for phase prediction. """

    def __init__(self, model_architecture, model_params, num_classes, gpu_id={}):
        """
        Arguments:
            model_architecture (list[str]) -- List of model types in order of desired hierarchy.
            model_params (dict) -- Dictionary of parameters to run / train model.
            num_classes (int) -- Number of classes.
        Keyword Arguments:
            gpu_id (list[int]) -- GPU number to use. if no GPU. (default:{})
        """
        self._model_architecture = model_architecture
        self._model_params = model_params
        self._num_classes = num_classes
        self._loss_func = torch.nn.BCEWithLogitsLoss()
        self._gpu_id = gpu_id

        if gpu_id:
            self._gpus = tf.config.experimental.list_physical_devices('GPU')
            if self._gpus:
                self._compute_mode = 'gpu'
            else:
                self._compute_mode = 'cpu'
        else:
            self._compute_mode = 'cpu'

    def _build_model(self, n_features):
        """Builds the ML models.
        Args:
            n_features (int): Number of features to learn on.
        Raises:
            RuntimeError: Error for incorrect model type.
        Returns:
            model: ML model to be returned.
        """
        if self._model_architecture == 'DNN':
            # Build a DNN model in Keras
            model = DNN(self._model_params)
            # else:
            with tf.device('/CPU:0'):
                model.define_model(n_features)
        elif self._model_architecture == 'XGB':
            # Build xgboost model
            model = XGB(self._model_params)
        elif self._model_architecture == 'GNN':
            # Build graph network with DGL
            print('GNN')
        else:
            raise RuntimeError(
                'Non standard architecture given. Must be "DNN", "XGB", "GNN", or comma-separated combination.')
        return model

    def training(self, x_train, y_train, x_test, y_test):
        """Trains the ML model.
        Args:
            x_train (np.array): Training data.
            y_train (np.array): Training labels.
            x_test (np.array): Test data.
            y_test (np.array): Test labels.
        Returns:
            train_loss: Training loss.
            test_loss: Test loss.
        """
        n_feats = x_train.shape[1]
        mod = self._build_model(n_feats)
        if 'lr' in self._model_params.keys():
            mod.train_model(x_train, y_train, x_test, y_test)
        else:
            mod.train_model(x_train, y_train, x_test, y_test)
        train_pred = mod.predict(x_train)
        test_pred = mod.predict(x_test)
        self._model = mod

        train_loss = roc_auc_score(y_train, train_pred)
        test_loss = roc_auc_score(y_test, test_pred)

        return train_loss, test_loss

    def cross_val_train(self, x, y, n_folds, shuffle=True):
        """Training with cross validation.
        Args:
            x (np.array): Feature data.
            y (np.array): Labels
            n_folds (int): Number of cross validation folds.
        Keyword Args:
            shuffle (bool): Flag to shuffle training data. Defaults to True.
        Returns:
            train_loss: Training loss.
            test_loss: Test loss.
        """
        n_feats = x.shape[1]
        kfold_splitter = KFold(n_splits=n_folds, shuffle=shuffle)
        train_test_inds = kfold_splitter.split(x)
        n_mods = n_folds

        self._models = [None for _ in range(n_mods)]
        train_loss = np.empty(n_mods)
        test_loss = np.empty(n_mods)

        for i, (train_inds, test_inds) in enumerate(train_test_inds):
            x_train, y_train = x[train_inds], y[train_inds]
            x_test, y_test = x[test_inds], y[test_inds]

            curr_model = self._build_model(n_feats)
            curr_model.train_model(x_train, y_train, x_test, y_test)

            train_pred = curr_model.predict(x_train)
            test_pred = curr_model.predict(x_test)

            train_loss[i] = roc_auc_score(y_train, train_pred)
            test_loss[i] = roc_auc_score(y_test, test_pred)

            self._models[i] = curr_model

        return train_loss, test_loss


class TwoStepModel:
    """ A class for supervised/semi-supervised hierarchical 2-step model for particle phase prediction."""

    def __init__(self, model_name):
        """
        Arguments:
            model_name (str): Name of trained supervised model for first step.
        """
        self._model_name = model_name
        self._model, self._feat_list, self._scaler, self._model_params = load_model(
            self._model_name)

    def __supervised_labels(self, file, partial_thresh=0.98):
        """Internal method to place initial labels from first step.
        Arguments:
            file (str): Simulation file to label.
        Keyword Arguments:
            partial_thresh (float): Confidence level needed to keep label (default: {0.98})
        Raises:
            RuntimeError: If missing features.
        """
        self._file_name = file

        # Load features
        df = pd.read_pickle(file)

        # initialize averaging over frames
        preds = np.zeros(len(df))

        # loop over feature list
        for i in range(len(df)):
            try:
                df_frame = df['Features'][i][self._feat_list]
            except:
                missing_feat = [
                    f for f in self._feat_list if f not in df['Features'][0].columns]
                raise RuntimeError(
                    '{} is missing features: {}'.format(file, missing_feat))

            if 'aggregate_features' in list(self._model_params.keys()):
                if self._model_params['aggregate_features'] == True:
                    G = nx.read_gpickle(file.replace(
                        '/features/', '/graphs/').replace('.pkl', '.gpkl'))
                    df_frame = aggregate_features(
                        df_frame, G, feat_list=self._feat_list)
                    self._feat_list = df_frame.columns

            # normalize data
            x_norm = self._scaler.transform(df_frame.values.astype(np.float32))
            self._features = x_norm

            # Perform xgboost prediction
            if 'xgboost' in str(type(self._model)):
                dpred = xgb.DMatrix(
                    self._features, feature_names=self._feat_list)
                y = self._model.predict(dpred)
            # Perform DNN prediction
            elif 'keras' in str(type(self._model)):
                y_array = self._model.predict(self._features)
                y = y_array[:, 1]

            # Save supervised predictions
            self.supervised_pred = np.copy(y)

            # Label based off threshold value
            y[y >= partial_thresh] = 1
            y[y <= (1-partial_thresh)] = 0
            y[(y > (1-partial_thresh)) & (y < partial_thresh)] = 2

        # Save labels for use in the mask
        self.supervised_labels = y

    @staticmethod
    def __generate_masks(labels, cutoff=0.8):
        """Internal method to generate training masks for semi-supervised problem.
        Arguments:
            labels (np.array[int]): List of labels from first step of model
        Keyword Arguments:
            cutoff (float): Train test split. (default: {0.8})
        Returns:
            train_mask (np.array): Array of particles to use in training.
            val_mask (np.array): Array of particles to use in validation.
        """
        # Pull confident labels
        inds = np.where(labels < 2)[0]

        # Instantiate mask arrays
        train_mask = np.zeros(len(labels))
        val_mask = np.zeros(len(labels))

        # Select training and validation indices
        train_inds = np.random.choice(
            inds, int(cutoff*len(inds)), replace=False).astype(int)
        val_inds = [ind for ind in inds if ind not in train_inds]

        # Fill masks
        train_mask[train_inds] = 1
        val_mask[val_inds] = 1

        return train_mask, val_mask

    def GNNmask(self, file, train_split=0.8, partial_label_threshold=0.98):
        """Construct and train the GNN
        Arguments:
            file (gsd_file): Simulation file
        Keyword Arguments:
            train_split (float): Train-test split (default: {0.8})
            partial_label_threshold (float): Confidence needed to keep label (default: {0.98})
        """
        # Generate labels using supervised model
        self.__supervised_labels(
            file=file, partial_thresh=partial_label_threshold)

        # Generate Train / Test masks
        self._train_mask, self._val_mask = self.__generate_masks(
            labels=self.supervised_labels, cutoff=train_split)

    def tuneParameters(self, file, train_params, model_params=None, default=True, verbose=0):
        """Use to tune hyperparameters
        Arguments:
            file (gsd_file): Simulation file
            train_params (dict): Parameters for training GNN
        Keyword Arguments:
            model_params (dict): Parameters for constructing GNN
            default (boolean): Toggle default GNN build.
        """
        # Build GNN
        if model_params == None or default is True:
            _GNN = GAT.defaultGAT(len(self._feat_list))
        else:
            _GNN = GAT(model_params=model_params)

        # Train GNN
        _GNN.trainGAT(file, self._features, np.round(self.supervised_pred),
                      train_params, self._train_mask, self._val_mask, verbose=verbose)

        # Predict GNN
        self.GNN_labels = _GNN.predictGNN().numpy()

        # Record Accuracy and Loss
        self.trainAcc = _GNN._train_acc
        self.valAcc = _GNN._val_acc
        self.loss = _GNN._loss


class DNN():
    def __init__(self, model_params):
        self._batch_size = model_params['batch_size']
        self._pred_batch = model_params['pred_batch']
        self._epochs = model_params['epochs']
        self._layers = model_params['layer_size']
        self._activation = model_params['activation']
        self._loss_func = model_params['loss_func']
        self._optimizer = model_params['optimizer']
        self._train_metrics = model_params['train_metrics']

        # Check for learning rate
        if 'lr' in model_params.keys():
            self._lr = model_params['lr']
        else:
            self._lr = 0.1

        # Check for early stopping criteria
        if 'patience' in model_params.keys():
            self._patience = model_params['patience']
        else:
            self._patience = self._epochs

        # Check if batch normalization used
        if 'batch_norm' in model_params.keys():
            self._batch_norm = model_params['batch_norm']
            if (len(self._batch_norm) != len(self._layers)) and (True in self._batch_norm):
                raise RuntimeWarning(
                    "Batch norm declared, but unclear on which layer.")
                raise RuntimeWarning(
                    "Assuming intended on first {} layers".format(len(self._batch_norm)))
                self._batch_norm += (len(self._layers) -
                                     len(self._batch_norm))*[False]

        # Check for dropout
        if 'dropout' in model_params.keys():
            self._dropout = model_params['dropout']
            if (len(self._dropout) != len(self._layers)) and (sum(self._dropout) > 0):
                raise RuntimeWarning(
                    "Dropout declared, but unclear on which layer.")
                raise RuntimeWarning(
                    "Assuming intended after first {} layers".format(len(self._dropout)))
                self._dropout += (len(self._layers)-len(self._dropout))*[0]

    def define_model(self, n_features):
        # Define input shape
        input_shape = n_features

        self._model = Sequential()

        # build layers
        for l, layer in enumerate(self._layers):
            if l == 0:
                self._model.add(Dense(layer, input_shape=(input_shape,)))
            else:
                self._model.add(Dense(layer))

            if hasattr(self, '_batch_norm'):
                if self._batch_norm[l] is True:
                    self._model.add(BatchNormalization())
                if self._activation[l] == 'LeakyReLU':
                    self._model.add(LeakyReLU(alpha=0.1))
                elif self._activation[l] == 'softmax':
                    self._model.add(Activation('softmax', dtype='float32'))
                else:
                    self._model.add(Activation(self._activation[l]))
            else:
                if self._activation[l] == 'LeakyReLU':
                    self._model.add(LeakyReLU(alpha=0.1))
                elif self._activation[l] == 'softmax':
                    self._model.add(Activation('softmax', dtype='float32'))
                else:
                    self._model.add(Activation(self._activation[l]))

            if hasattr(self, '_dropout'):
                if self._dropout[l] != 0:
                    self._model.add(Dropout(self._dropout[l]))

        # Compile
        opt = tf.keras.optimizers.Adam(learning_rate=self._lr)
        self._model.compile(loss=self._loss_func,
                            optimizer=opt,
                            metrics=[self._train_metrics])

    def train_model(self, x_train, y_train, x_val, y_val):

        # Define Earlystopping criteria
        callbacks_list = [EarlyStopping(
            monitor='val_loss', patience=self._patience, restore_best_weights=True)]

        # One-Hot Encode labels
        if np.shape(y_train)[1] != self._layers[-1]:
            y_train = tf.keras.utils.to_categorical(y_train)

        if np.shape(y_val)[1] != self._layers[-1]:
            y_val = tf.keras.utils.to_categorical(y_val)

        # Train
        self.history = self._model.fit(x=x_train, y=y_train, epochs=self._epochs, batch_size=self._batch_size,
                                       verbose=0, validation_data=(x_val, y_val), callbacks=callbacks_list,
                                       )

        # Identify early stopping epoch
        self._early_stopping_epoch = callbacks_list[0].stopped_epoch

    def predict(self, x):
        # Predict from the model
        preds = self._model.predict(x, batch_size=self._pred_batch)

        # Return Predictions
        return preds

    def clear_model(self):
        # Clear the model
        clear_session()

        del(self._model)
        _ = gc.collect()

    ### Create Properties of class ###
    @property
    def model(self):
        return self._model

    @property
    def loss_func(self):
        return self._loss_func

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def metrics(self):
        return self._train_metrics

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def pred_batch(self):
        return self._pred_batch


class XGB():
    def __init__(self, model_params):
        # Set model parameters
        self._model_params = {}
        for param_key in model_params.keys():
            if param_key not in ['num_boost_round', 'early_stopping_rounds']:
                self._model_params[param_key] = model_params[param_key]
        # self._model_params = model_params

        self._early_stopping_rounds = model_params['early_stopping_rounds']
        self._num_boost_rounds = model_params['num_boost_round']

    def train_model(self, x_train, y_train, x_val, y_val):
        """Training step for XGB model.
        Args:
            x_train (numpy array): Training particle feature data
            y_train (numpy array): Training particle labels, not one-hot encoded.(unless multi:softprob used)
            x_val (numpy array): Validation particle feature data
            y_val (numpy array): Validation particle labels. Same shape as y_train
        """
        # Make DMatrices
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)

        # Eval list
        eval_list = [(dtrain, 'train'), (dval, 'eval')]

        # Train
        self._model = xgb.train(params=self._model_params, dtrain=dtrain,
                                num_boost_round=self._num_boost_rounds, evals=eval_list,
                                early_stopping_rounds=self._early_stopping_rounds,
                                verbose_eval=False)

        self._early_stopping_epoch = self._model.best_ntree_limit

    def predict(self, x):
        # Prediction DMatrix
        x = xgb.DMatrix(x)

        # Predict
        den_preds = self._model.predict(
            x, ntree_limit=self._early_stopping_epoch)
        gas_preds = np.ones(len(den_preds)) - den_preds
        preds = np.vstack((gas_preds, den_preds)).T
        return den_preds

    @staticmethod
    def get_probabilities(den_preds):
        gas_preds = np.ones(len(den_preds)) - den_preds
        preds = np.vstack((gas_preds, den_preds)).T
        return preds

    @property
    def model_params(self):
        return self._model_params

    @property
    def epochs(self):
        return self._num_boost_rounds

    @property
    def early_stopping_epoch(self):
        return self._early_stopping_epoch

# Define class for GNN model.


class GAT():
    def __init__(self, model_params):
        self._model_params = model_params

        # Instantiate graph
        G = dgl.DGLGraph()

        # Build Model
        self._GAT = TorchGAT(G, self._model_params['layers'], self._model_params['n_features'],
                             self._model_params['hidden_features'], self._model_params['n_classes'],
                             self._model_params['attention_heads'], self._model_params['activation'],
                             self._model_params['feat_drop'], self._model_params['attn_drop'],
                             self._model_params['negative_slope'], self._model_params['residual'])

    def trainGAT(self, file_name, features, labels, train_params, train_mask=None, val_mask=None, verbose=0):
        """Internal method to train the GNN for each graph.
        Arguments:
            file_name {str} -- Simulation file of interest.
            features {np.array} -- Normalized features.
            labels {np.array} -- Supervised labels.
            train_params {dict} -- Training parameters.
            train_mask {array} -- Mask for training nodes.
            val_mask {array} -- Mask for validation nodes.
            verbose {int} -- Determine whether to print or not.
        """
        self._train_params = train_params

        # Load graph
        g = nx.read_gpickle(file_name.replace(
            'features', 'graphs').replace('.pkl', '.gpkl'))
        g.remove_edges_from(nx.selfloop_edges(g))
        G = dgl.DGLGraph()
        G.from_networkx(g)
        self._GAT.g = G

        # Write features and labels in Torch tensor form
        self._features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        # put masks in Torch tensor form
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(train_mask)
            val_mask = torch.BoolTensor(val_mask)
        else:
            train_mask = torch.ByteTensor(train_mask)
            val_mask = torch.ByteTensor(val_mask)

        # Allocate GPU space for training
        if self._train_params['gpu'] < 0:
            cuda = False
        else:
            cuda = True
            torch.cuda.set_device(self._train_params['gpu'])
            self._features = self._features.cuda()
            labels = labels.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            self._GAT.cuda()

        # Define Loss and Optimizer
        loss_fcn = torch.nn.CrossEntropyLoss()
        # loss_fcn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self._GAT.parameters(
        ), lr=self._train_params['lr'], weight_decay=self._train_params['weight_decay'])

        # Set accuracy checking
        timestep = 0
        self._train_acc = []
        self._val_acc = []
        self._loss = []

        for epoch in range(self._train_params['epochs']):
            self._GAT.train()

            # forward
            logits = self._GAT(self._features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # self._GAT.to("cpu")
            if timestep % 20 == 0:
                train_acc = self._accuracy(
                    logits[train_mask], labels[train_mask])

                val_acc = self._evaluate(
                    self._GAT, self._features, labels, val_mask)

                if verbose > 0:
                    print("Epoch {:05d} |  Loss {:.4f} | TrainAcc {:.4f} |"
                          " ValAcc {:.4f} ".format(epoch, loss.item(), train_acc,
                                                   val_acc))

                self._train_acc.append(train_acc)
                self._val_acc.append(val_acc)
                self._loss.append(loss.item())
            timestep += 1
        if verbose > 0:
            print("\n")

    def predictGNN(self):
        """Internal method for GNN prediction
        Arguments:
            features {np.array} -- Normalized features.
        Returns:
            {array(int)} -- Particle labels
        """
        with torch.no_grad():
            self._GAT.eval()

            GAT_labels = self._GAT(self._features)

            _, test_labels = torch.max(GAT_labels, dim=1)

        return test_labels.cpu()

    @staticmethod
    def _accuracy(logits, labels):
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    def _evaluate(self, model, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(features)
            logits = logits[mask]
            labels = labels[mask]
            return self._accuracy(logits, labels)

    @classmethod
    def defaultGAT(cls, n_features):
        model_params = {'layers': 2, 'n_features': n_features, 'hidden_features': [8, 8], 'n_classes': 2,
                        'attention_heads': [4, 4, 1], 'activation': F.elu, 'feat_drop': 0, 'attn_drop': 0,
                        'negative_slope': 0.2, 'residual': True}
        return cls(model_params)
