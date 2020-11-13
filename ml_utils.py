"""
Utilities package to better automate the ML process for phase identification.
"""
# General
import glob
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

# ML Tools
from ml_features import aggregate_features
from TorchModels import TorchGAT, train_GAT
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import torch
import sklearn
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

# Image tools
from image_labeling import image_labeling
import gsd.hoomd

# Graph Packages
import networkx as nx
from dgl import DGLGraph


###################
### ML Training ###
###################


def split_files(filepath="../features/"):
    """Reads in pickle data files and splits into training and test files based on phase diagram cutoffs.
    Keyword Args:
        filepath (str): Path to feature data.
    Returns:
        other_files (list[str]): List of simulation files not used in training.
        train_files (list[str]]): List of simulation files used for training.
    """
    # Read in all feature files
    all_files = glob.glob(filepath + "*.pkl")

    # Remove the path
    all_files = [all_files[i].split("/")[-1] for i in range(len(all_files))]

    train_files = []
    for file in all_files:
        phi = float(file.split("_")[1])
        t = float(file.split("_")[3])

        if (
            ((t > 12) and (phi >= 0.78))
            or ((t > 12) and (phi <= 0.32))
        ):
            train_files.append(file)
    # Define other files
    other_files = [x for x in all_files if x not in train_files]

    return other_files, train_files


def label(file, features, path="../features/", feat_agg=False, all_frames=False, random_state=None):
    """Labels particles as dilute (0), interface (1), or dense (1) based on position in the phase diagram 
    and certain order parameter values.
    Args:
        file (str): Name of feature file.
        features (list[str]): List of feature names to use for training.
    Keyword Args:
        path (str): Path to all feature data.
        feat_agg (bool): Flag to determine whether feature aggregation will be performed or not. Defaults to False.
        all_frames (bool): Flag to determine if all frames are to be processed. Defaults to False.
        random_state (int): Random state to use for sampling. Defaults to None.
    Returns:
        df_tmp (pd.DataFrame): Labeled feature data frame.
    """
    # Read in file
    phi = float(file.split("_")[1])
    t = float(file.split("_")[3])

    # Load feature object
    if all_frames == False:
        df_tmp = pd.read_pickle(path+file)['Features'][-1:]

        # Aggregate features if necessary
        if feat_agg == True:
            df_G = pd.read_pickle(path.replace('features', 'graphs')+file)
    else:
        df_tmp = pd.read_pickle(path+file)['Features'][:]

        # Aggregate features if necessary
        if feat_agg == True:
            df_G = pd.read_pickle(path.replace('features', 'graphs')+file)[-1:]

    for i, df in enumerate(df_tmp):
        # Add aggregate features
        if feat_agg == True:
            G = df_G['Features'].iloc[i]
            df = aggregate_features(df, G, features)

        pd.set_option("mode.chained_assignment", None)
        # Label dilute files
        if ((phi < 0.18) & (t <= 6)) or ((phi < 0.33) and (t < 2) or ((phi <= 0.32) and (t > 12))):
            df["label"] = np.zeros(len(df))
            if random_state is not None:
                df = df.sample(frac=0.07, random_state=random_state)
            else:
                df = df.sample(frac=0.07)

        # Label dense files
        elif ((phi >= 0.75) and (t < 4)) or ((phi >= 0.78) and (t > 12)):
            df["label"] = np.ones(len(df))
        df_tmp[i] = df

    return df_tmp


def image_data(files, features, feat_agg=False):
    """ Generates training data for phase separated systems based on images.
    Args:
        files (list[str]): List of trajectory files
        features (list[str]): List of features.
    Keyword Args:
        feat_agg (bool): Boolean for feature aggregation. Defaults to False.
    Returns:
        x_tot (np.array): Array of training data.
        y_tot (np.array): Array of training labels.
    """
    for j, file in enumerate(files):
        frame = -1
        im_name = file.replace(".gsd", ".tif").replace(
            "/simulations/", "/snapshots/")

        # Generate indices for figure. Make figure if it doesn't exist
        dil_inds, den_inds = image_labeling(im_name, file, frame=frame)

        # Load feature
        feat_path = file.replace(".gsd", ".pkl").replace(
            "/simulations/", "/features/")
        df = pd.read_pickle(feat_path)
        if len(df) == 1:
            df = df['Features'][0][features]
        else:
            df = df['Features'][frame][features]
        # Make sure training data is balanced
        if len(den_inds) > len(dil_inds):
            sub_den_inds = np.random.choice(
                den_inds, len(dil_inds), replace=False)
            sub_dil_inds = dil_inds
        else:
            sub_dil_inds = np.random.choice(
                dil_inds, len(den_inds), replace=False)
            sub_den_inds = den_inds

        pd.set_option("mode.chained_assignment", None)
        if feat_agg == True:
            G = nx.read_gpickle(feat_path.replace(
                'features', 'graphs').replace('.pkl', '.gpkl'))
            df = aggregate_features(df, G, features)

        # Fix labels
        df['label'] = 2*np.ones(len(df))
        df['label'].iloc[sub_den_inds] = 1
        df['label'].iloc[sub_dil_inds] = 0

        # Append features to total df
        df = df.iloc[sub_den_inds.tolist()+sub_dil_inds.tolist()]
        if j is 0:
            x_tot = df.drop(
                labels='label', axis=1).to_numpy().astype('float64')
            y_tot = df.label.to_numpy().astype('float64')
        else:
            x_tot = np.append(x_tot, df.drop(
                labels='label', axis=1).to_numpy().astype('float64'), axis=0)
            y_tot = np.append(
                y_tot, df.label.to_numpy().astype('float64'), axis=0)

    return x_tot, y_tot


def get_training_data(features, filepath="../features/", feat_agg=False, rand_train=True,
                      random_state=None, all_times=False, return_df=False):
    """Parses through all feature data to determine files for training and loads training data.
    Args:
        features (list[str]): List of features to use for training.
    Keyword Args:
        filepath (str): Path to feature data.
        feat_agg (bool): Flag for aggregating 1st neighbor features or not. Defaults to False.
        rand_train (bool): Flag for randomly training or not. Defaults to True.
        random_state (int): Random state for training repeatability. Defaults to None.
        all_times (bool): Flag to get data at all time steps or not. Defaults to False.
        return_df (bool): Flag to return full data frame or not. Defaults to False.
    """
    # Load and shuffle training files
    _, train_files = split_files(filepath=filepath)

    if rand_train is True:
        np.random.shuffle(train_files)

    # Iterate through training files
    for j, file in enumerate(train_files):
        file_name = file.split("/")[-1]
        df_tmp = label(file_name, features, path=filepath, feat_agg=feat_agg,
                       all_frames=all_times, random_state=random_state)

        # Determine which times to search
        if all_times == False:
            df_times = df_tmp.iloc[-1:]
        else:
            df_times = df_tmp.iloc[:]

        for i, df in enumerate(df_times):
            # Initialize training data
            if j is 0 and i is 0:
                x_tot = df.drop(
                    labels="label", axis=1).to_numpy().astype("float64")
                y_tot = df.label.to_numpy().astype("float64")

                if return_df == True:
                    df_tot = df

            # Add to training data array
            else:
                x_tot = np.append(
                    x_tot,
                    df.drop(labels="label", axis=1).to_numpy().astype(
                        "float64"),
                    axis=0,
                )
                y_tot = np.append(
                    y_tot, df.label.to_numpy().astype("float64"), axis=0)

                if return_df == True:
                    df_tot = df_tot.append(df)
    if return_df == False:
        return x_tot, y_tot, df.columns
    else:
        return x_tot, y_tot, df_tot


def save_model(model, scaler, feature_list, model_params, number, model_path):
    """Saves the model and all relevant parameters.
    Args:
        model : ML model itself.
        scaler (scikit-learn.transformer): Transformer for feature normalization.
        feature_list (list[str]): List of features used for training.
        model_params (dict): Dictionary of parameters used for the model.
        number (int): Used to distinguish between various model iterations.
        model_path (str): Path of where to save model.
    Raises:
        Exception: Warns when you are attempting to overwrite a previously saved model.
    """
    if 'xgb' in str(type(model)):
        model_type = 'xgb'
        tag = '.model'
    elif 'keras' in str(type(model)):
        model_type = 'dnn'
        tag = '.h5'
    else:
        model_type = 'gnn'
        tag = '.pt'
    model_name = "_".join([model_type, str(number), tag])
    if os.path.isfile(model_path+model_name):
        raise Exception("You already have a model called this!")
    else:
        model.save_model(model_path+model_name+'.model')
        pickle.dump(scaler, open(model_path+model_name+'scaler.pkl', 'wb'))
        np.save(model_path+model_name+"features.npy", np.array(feature_list))
        np.save(model_path+model_name+'param.npy', model_params)
    return

##########################
### WIDOM CALCULATIONS ###
##########################


def load_model(model_name):
    """Loads a saved model and all relevant objects.
    Args:
        model_name (str): Saved model path and name.
    Returns:
        model: Trained ML model.
        feat_list (list(str)): List of features used to train the model.
        model_scaler (scikit-learn.transformer): Fit scaler used for the normalization of data in training.
        model_params (dict): Dictionary of model parameters.
    """
    model_type = model_name.split('/')[-1].split('_')[-4]
    feat_list = np.load(model_name+'features.npy').tolist()
    model_params = np.load(model_name+'param.npy', allow_pickle=True).item()
    model_scaler = pickle.load(open(model_name+'scaler.pkl', 'rb'))

    if model_type == 'xgb':
        model = xgb.Booster()
        model.load_model(model_name+'.model')
    elif model_type == 'dnn':
        model = tf.keras.models.load_model(model_name+'.h5')
    elif model_type == 'gnn':
        g = DGLGraph()
        model = TorchGAT(g, model_params['layers'], model_params['n_features'], model_params['hidden_features'],
                         model_params['n_classes'], model_params['attention_heads'], model_params['activation'],
                         model_params['feat_drop'], model_params['attn_drop'], model_params['negative_slope'],
                         model_params['residual'])
        model.load_state_dict(torch.load(model_name+'.pt'))

    return model, feat_list, model_scaler, model_params


def gas_fraction(file, model, feat_list, scaler, model_params, partial_pred=False, partial_thresh=0.98):
    """Computes the fraction of gas particles in a simulation snapshot based on predictions.
    Args:
        file (str): File name.
        model : Trained ML model.
        feat_list (list[str]): List of features for prediction.
        scaler (scikit-learn.transformer): Fit transformer for normalizing features.
        model_params (dict): Dictionary of model parameters.
    Keyword Args:
        partial_pred (bool): Flag indicating whether to compute gas fraction from only confident predictions. Defaults to False.
        partial_thresh (float): Confidence threshold for partial predictions. Defaults to 0.98.
    Raises:
        RuntimeError: Warns that necessary features are missing from the feature data frame.
    Returns:
        gas_frac (float): Fraction of gas phase particles.
    """
    df = pd.read_pickle(file)

    if 'aggregate_features' in list(model_params.keys()):
        if model_params['aggregate_features'] == True:
            if len(df) > 1:
                df_G = pd.read_pickle(file.replace('features', 'graphs'))
            else:
                df_G = nx.read_gpickle(file.replace(
                    '/features/', '/graphs/').replace('.pkl', '.gpkl'))

    # initialize averaging over frames
    preds = np.zeros(len(df))

    # loop over feature list
    for i in range(len(df)):
        try:
            df_frame = df['Features'][i][feat_list]
        except:
            missing_feat = [
                f for f in feat_list if f not in df['Features'][0].columns]
            raise RuntimeError(
                '{} is missing features: {}'.format(file, missing_feat))
        if len(df) > 1:
            G = df_G['Features'][i]
        else:
            G = df_G
        df_frame = aggregate_features(df_frame, G, feat_list=feat_list)

        # normalize data
        x_norm = scaler.transform(df_frame.values.astype(np.float32))

        # Perform xgboost prediction
        if 'xgboost' in str(type(model)):
            dpred = xgb.DMatrix(x_norm, feature_names=df_frame.columns)
            y = model.predict(dpred)

            if partial_pred == True:
                y[y >= partial_thresh] = 1
                y[y <= (1-partial_thresh)] = 0
                y[(y > (1-partial_thresh)) & (y < partial_thresh)] = 2
        # Perform DNN prediction
        elif 'keras' in str(type(model)):
            y_array = model.predict(x_norm)

            if partial_pred == False:
                y = np.argmax(y_array, axis=1)
            elif partial_pred == True:
                y_val = np.max(y_array, axis=1)
                y = np.argmax(y_array, axis=1)
                y[y_val < partial_thresh] = 2

        # Perform GNN prediction
        else:
            torch_features = torch.FloatTensor(x_norm)
            if len(df) == 0:
                g = nx.read_gpickle(file.replace(
                    'features/', 'graphs/').replace('.pkl', '.gpkl'))
            else:
                g = nx.read_gpickle(file.replace(
                    'features/', 'graphs/').replace('.pkl', '.gpkl'))
            g.remove_edges_from(nx.selfloop_edges(g))
            G = DGLGraph()
            G.from_networkx(g)

            y = model.predict(torch_features, G)
            m = torch.nn.Softmax(dim=1)
            y = m(y.data.cpu())
            # _, y = torch.max(y,  dim=1)
            if partial_pred == False:
                y = np.argmax(y.numpy(), axis=1)
            elif partial_pred == True:
                y_val = np.max(y.numpy(), axis=1)
                y = np.argmax(y.numpy(), axis=1)
                y[y_val < partial_thresh] = 2

        # Append prediction
        if partial_pred == False:
            preds[i] = len(np.where(y.round() == 0.)[0]) / len(y)
        else:
            preds[i] = len(y[y == 0]) / len(y[y != 2])

    gas_frac = np.mean(preds)

    return gas_frac


def two_step_gas_fraction(file, model, feat_list, scaler, model_params, partial_pred=False, partial_thresh=0.98):
    """Computes the fraction of gas particles in a simulation snapshot based on predictions from the two-step model framework.
    Args:
        file (str): File name.
        model : Trained ML model.
        feat_list (list[str]): List of features for prediction.
        scaler (scikit-learn.transformer): Fit transformer for normalizing features.
        model_params (dict): Dictionary of model parameters.
    Keyword Args:
        partial_pred (bool): Flag indicating whether to compute gas fraction from only confident predictions. Defaults to False.
        partial_thresh (float): Confidence threshold for partial predictions. Defaults to 0.98.
    Raises:
        RuntimeError: Warns that necessary features are missing from the feature data frame.
    Returns:
        gas_frac (float): Fraction of gas phase particles.
    """
    df = pd.read_pickle(file)

    if 'aggregate_features' in list(model_params.keys()):
        if model_params['aggregate_features'] == True:
            if len(df) > 1:
                df_G = pd.read_pickle(file.replace('features', 'graphs'))
            else:
                df_G = nx.read_gpickle(file.replace(
                    '/features/', '/graphs/').replace('.pkl', '.gpkl'))

    # initialize averaging over frames
    preds = np.zeros(len(df))

    # loop over feature list
    for i in range(len(df)):
        try:
            df_frame = df['Features'][i][feat_list]
        except:
            missing_feat = [
                f for f in feat_list if f not in df['Features'][0].columns]
            raise RuntimeError(
                '{} is missing features: {}'.format(file, missing_feat))
        if len(df) > 1:
            G = df_G['Features'][i]
        else:
            G = df_G

        df_frame = aggregate_features(df_frame, G, feat_list=feat_list)

        # normalize data
        x_norm = scaler.transform(df_frame.values.astype(np.float32))

        # Perform xgboost prediction
        if 'xgboost' in str(type(model)):
            dpred = xgb.DMatrix(x_norm, feature_names=df_frame.columns)
            y = model.predict(dpred)

            if partial_pred == True:
                y[y >= partial_thresh] = 1
                y[y <= (1-partial_thresh)] = 0
                y[(y > (1-partial_thresh)) & (y < partial_thresh)] = 2
        # Perform DNN prediction
        elif 'keras' in str(type(model)):
            y_array = model.predict(x_norm)
            y_val = np.max(y_array, axis=1)
            y = np.argmax(y_array, axis=1)
            # y[y_val < partial_thresh] = 2

        # Perform GNN prediction
        val_mask = np.zeros(len(y))
        train_mask = np.zeros(len(y))

        conf_thresh = 0.98
        inds = np.argwhere((y_val >= conf_thresh)).flatten()
        while len(inds) < 500:
            conf_thresh -= 0.02
            inds = np.argwhere((y_val >= conf_thresh)).flatten()
            if len(inds) > 500:
                if conf_thresh < 0.9:
                    break
            else:
                if conf_thresh < 0.8:
                    break
        print(inds.shape)
        t_inds = np.random.choice(
            inds, int(0.6*len(inds)), replace=False).astype(int)
        v_inds = [ind for ind in inds if ind not in t_inds]

        train_mask[t_inds] = 1
        val_mask[v_inds] = 1
        print('Mask Sums: ', sum(train_mask), sum(val_mask))
        gnn_model = train_GAT(x_norm, train_mask, val_mask, y, G)

        G2 = DGLGraph()
        G2.from_networkx(G)

        y = gnn_model(torch.FloatTensor(x_norm), G2)
        m = torch.nn.Softmax(dim=1)
        y = m(y.data.cpu())

        # Average the first and second step predictions.
        y = (y+torch.FloatTensor(y_array))/2
        # _, y = torch.max(y,  dim=1)
        if partial_pred == False:
            y = np.argmax(y.numpy(), axis=1)
        elif partial_pred == True:
            y_val = np.max(y.numpy(), axis=1)
            y = np.argmax(y.numpy(), axis=1)
            y[y_val < partial_thresh] = 2

        # Append prediction
        if partial_pred == False:
            preds[i] = len(np.where(y.round() == 0.)[0]) / len(y)
        else:
            preds[i] = len(y[y == 0]) / len(y[y != 2])

    gas_frac = np.mean(preds)

    return gas_frac


def widom_calc(file_location, model_name, partial_pred=False, partial_thresh=0.98):
    """Returns the gas fraction at each point in the phase diagram above the critical point.
    Args:
        file_location (str): Path to feature files
        model_name (str): Name of trained model to use for prediction
    Keyword Args:
        partial_pred (bool): Flag for only confident predictions. Defaults to False.
        partial_thresh (float): Confidence threshold for partial predictions. Defaults to 0.98.
    Returns:
        df_widom (pd.DataFrame): Data frame of phase location and gas fraction for each simulation in file_location.
    """
    # Read files
    all_files = glob.glob(file_location+'*.pkl')
    all_files = [file for file in all_files if 12 > float(file.split('_')[-2])]

    # Load model parameters
    the_model, feat_list, scaler, params = load_model(model_name)

    df_widom = pd.DataFrame(np.zeros((len(all_files), 3)))
    df_widom.columns = ['phi', 'tau', 'gas_frac']

    for i, file in enumerate(tqdm(all_files)):
        df_widom['phi'][i] = float(file.split('/')[-1].split('_')[1])
        df_widom['tau'][i] = float(file.split('/')[-1].split('_')[3])
        df_widom['gas_frac'][i] = gas_fraction(
            file, the_model, feat_list, scaler, params, partial_pred=partial_pred, partial_thresh=partial_thresh)
    print("Finished Training")
    if 'keras' in str(type(the_model)):
        clear_session()
        del(the_model)

    return df_widom


def coexist_calc(file_location, model_name, partial_pred=False, partial_thresh=0.98):
    """Returns the gas fraction at each point in the phase diagram below the critical point as predicted by single step model.
    Args:
        file_location (str): Path to feature files
        model_name (str): Name of trained model to use for prediction
    Keyword Args:
        partial_pred (bool): Flag for only confident predictions. Defaults to False.
        partial_thresh (float): Confidence threshold for partial predictions. Defaults to 0.98.
    Returns:
        df_widom (pd.DataFrame): Data frame of phase location and gas fraction for each simulation in file_location.
    """
    # Read files
    all_files = glob.glob(file_location+'*.pkl')
    all_files = [file for file in all_files if 12 <=
                 float(file.split('_')[-2])]

    # Load model parameters
    the_model, feat_list, scaler, params = load_model(model_name)

    df_coexist = pd.DataFrame(np.zeros((len(all_files), 3)))
    df_coexist.columns = ['phi', 'tau', 'gas_frac']

    for i, file in enumerate(tqdm(all_files)):
        # print('Running File {}'.format(i))
        df_coexist['phi'][i] = float(file.split('/')[-1].split('_')[1])
        df_coexist['tau'][i] = float(file.split('/')[-1].split('_')[3])
        df_coexist['gas_frac'][i] = gas_fraction(
            file, the_model, feat_list, scaler, params, partial_pred=partial_pred, partial_thresh=partial_thresh)
    print("Finished Training")
    if 'keras' in str(type(the_model)):
        clear_session()
        del(the_model)

    return df_coexist


def gnn_coexist_calc(file_location, model_name, partial_pred=False, partial_thresh=0.98):
    """Returns the gas fraction at each point in the phase diagram below the critical point as predicted by the two-step model.
    Args:
        file_location (str): Path to feature files
        model_name (str): Name of trained model to use for prediction
    Keyword Args:
        partial_pred (bool): Flag for only confident predictions. Defaults to False.
        partial_thresh (float): Confidence threshold for partial predictions. Defaults to 0.98.
    Returns:
        df_widom (pd.DataFrame): Data frame of phase location and gas fraction for each simulation in file_location.
    """
    # Read files
    all_files = glob.glob(file_location+'*.pkl')
    all_files = [file for file in all_files if 12 <=
                 float(file.split('_')[-2])]

    # Load model parameters
    the_model, feat_list, scaler, params = load_model(model_name)
    df_coexist = pd.DataFrame(np.zeros((len(all_files), 3)))
    df_coexist.columns = ['phi', 'tau', 'gas_frac']

    for i, file in enumerate(tqdm(all_files)):
        # print('Running File {}'.format(i))
        df_coexist['phi'][i] = float(file.split('/')[-1].split('_')[1])
        df_coexist['tau'][i] = float(file.split('/')[-1].split('_')[3])
        df_coexist['gas_frac'][i] = two_step_gas_fraction(
            file, the_model, feat_list, scaler, params, partial_pred=partial_pred, partial_thresh=partial_thresh)
    print("Finished Training")
    if 'keras' in str(type(the_model)):
        clear_session()
        del(the_model)

    return df_coexist


def model_predict(df_feats, file, model_type, model_path, model_num, classes=2, partial_thresh=0.98):
    """Returns particle labels.

    Args:
        df_feats (pd.DataFrame): Feature data frame.
        file (str): Name of feature file.
        model_type (str): Model type (XGB, DNN, or GNN)
        model_path (str): Path to model.
        model_num (int): Specific model id number.
    Keyword Args:
        classes (int): Number of classes to predict. Defaults to 2.
        partial_thresh (float): Confidence threshold for initial labels used for training the GNN. Defaults to 0.98.
    Returns:
        y (np.array): Array of particle labels.
    """
    if classes == 2:
        num_classes = "binary"
    else:
        num_classes = "multi" + str(classes)

    model_name = "{}_{}_{}_".format(model_type, num_classes, str(model_num))
    model, feat_list, scaler, model_params = load_model(model_name)

    ### Run prediction ###
    df_pred = df_feats[feat_list].copy()
    if "aggregate_features" in list(model_params.keys()):
        if model_params['aggregate_features'] == True:
            G = nx.read_gpickle(file.replace(
                "features/", "graphs/").replace('pkl', 'gpkl'))
            df_pred = aggregate_features(df_pred, G, feat_list=feat_list)
    x = df_pred.values.astype(np.float32)

    # Scale Data
    x_norm = scaler.transform(x)

    # Neural Net
    if model_type == "dnn":
        y_array = model.predict(x_norm)
        y = y_array[:, 1]

    # XGB Random Forest
    elif model_type == "xgb":
        dpred = xgb.DMatrix(x_norm, feature_names=df_pred.columns)
        y = model.predict(dpred)

    # GAT GNN
    elif model_type == "gnn":
        torch_features = torch.FloatTensor(x_norm)
        g = nx.read_gpickle("../graphs/" + file + ".gpkl")
        g.remove_edges_from(nx.selfloop_edges(g))
        G = DGLGraph()
        G.from_networkx(g)
        with torch.no_grad():
            model.eval()
            model.g = G
            for layer in model.gat_layers:
                layer.g = G
            output = model(torch_features.float())
            y_array = output.data.cpu()
            m = torch.nn.Softmax(dim=1)
            y_array = m(y_array).numpy()
            y = y_array[:, 1]

    ### Coloring Based on Confidence of Prediction ###
    y[y >= partial_thresh] = 1
    y[y <= (1-partial_thresh)] = 0
    y[(y > (1-partial_thresh)) & (y < partial_thresh)] = 2

    return y


######################
### VISUALIZATIONS ###
######################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    with sns.axes_style('dark'):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=20)
        plt.yticks(tick_marks, classes, fontsize=20)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize == True:
                plt.text(j, np.abs(i), format(cm[i, j], fmt),
                         horizontalalignment="center", verticalalignment='center',
                         color="white" if cm[i, j] > thresh else "black", fontsize=20)
            else:
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center", verticalalignment='center',
                         color="white" if cm[i, j] > thresh else "black", fontsize=20)

        plt.tight_layout()
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
