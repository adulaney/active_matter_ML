import numpy as np
from functools import partial
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe
from ml_models import PhasePredictModel


def ml_hyperopt(model_type, model_params, x_train, y_train, x_val, y_val, opt_type='full', x_train_ps=None,
                y_train_ps=None, x_val_ps=None, y_val_ps=None, phases=2, gpu_id=[],
                n_cv=5, hyperopt_rounds=100, normalizer=False, train_params={}, opt_params=None):

    if opt_params != None:
        search_space_names = opt_params
    else:
        search_space_names = list(model_param_space[model_type][opt_type])

    search_space = [search_spaces[model_type][search_var]
                    for search_var in search_space_names]
    search_space_dict = {name: var_space for name, var_space
                         in zip(search_space_names, search_space)}

    if opt_type == 'regularization' and model_type == "DNN":
        n_layers = len(model_params['layer_size'])
        search_space_dict = regularizer_space_helper(
            search_space_dict, n_layers)

    optimizer_kwargs = {"mod_params": model_params, "train_params": train_params,
                        "model_type": model_type,
                        "x_train": x_train, "y_oh_train": y_train,
                        "x_val": x_val, "y_oh_val": y_val,
                        "x_train_ps": x_train_ps, "y_oh_train_ps": y_train_ps,
                        "x_val_ps": x_val_ps, "y_oh_val_ps": y_val_ps,
                        "phases": phases, "gpu_id": gpu_id, "n_cv": n_cv}

    complete_opt_helper = partial(opt_helper, **optimizer_kwargs)

    trials = Trials()
    global history
    history = []

    fmin(complete_opt_helper, search_space_dict, algo=tpe.suggest, max_evals=hyperopt_rounds,
         trials=trials)

    # sort based on loss
    history = sorted(history, key=lambda tpl: tpl[1])

    return history


def condense_DNN_space(space, opt_type='full'):
    if opt_type == 'full':
        adj_params = {'activation': [], 'layer': [],
                      'dropout': [], 'batchnorm': []}

    elif opt_type == 'architecture':
        adj_params = {'activation': [], 'layer': []}

    else:
        adj_params = {'dropout': [], 'batchnorm': []}

    for key in sorted(space.keys()):
        key_base = key.split("_")[0]
        if key_base in adj_params.keys():
            adj_params[key_base].append(space[key])
            space.pop(key)

    # Reconstruct the correct keys
    if 'layer' in adj_params.keys() and len(adj_params['layer']) > 1:
        space['layer_size'] = adj_params['layer'] + [2]
    if 'activation' in adj_params.keys() and len(adj_params['activation']):
        space['activation'] = adj_params['activation'] + ['softmax']
    if 'dropout' in adj_params.keys() and len(adj_params['dropout']):
        space['dropout'] = adj_params['dropout'] + [0]
    if 'batchnorm' in adj_params.keys() and len(adj_params['dropout']):
        space['batch_norm'] = adj_params['batchnorm'] + [False]


def space_to_params(space, model_params, model_type, opt_type='full'):
    new_params = model_params
    if model_type == 'DNN':
        condense_DNN_space(space, opt_type=opt_type)
    for key in space.keys():
        new_params[key] = space[key]

    return new_params


def opt_helper(space, mod_params=None, train_params=None, model_type=None, x_train=None, y_oh_train=None,
               x_train_ps=None, y_oh_train_ps=None, x_val=None, y_oh_val=None,
               x_val_ps=None, y_oh_val_ps=None, phases=2, gpu_id=-1, n_cv=0):
    """ Used to optimize DNN architecture using hyperopt package.

    Args:
        space (tuple): Parameter space
    """
    if "normalizer" in space.keys():
        scaler = transformers[space['normalizer']]
        scaler.fit(np.concatenate([x_train, x_train_ps]))

        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_train_ps = scaler.transform(x_train_ps)
        x_val_ps = scaler.transform(x_val_ps)

    elif 'normalizer' in mod_params.keys():
        scaler = transformers[mod_params['normalizer']]
        scaler.fit(np.concatenate([x_train, x_train_ps]))

        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_train_ps = scaler.transform(x_train_ps)
        x_val_ps = scaler.transform(x_val_ps)

    model_params = space_to_params(space, mod_params, model_type)

    if n_cv <= 2 and x_train_ps is not None:
        mod_hom = PhasePredictModel(
            model_type, model_params, phases, gpu_id=gpu_id)
        mod_sep = PhasePredictModel(
            model_type, model_params, phases, gpu_id=gpu_id)

        _, test_hom = mod_hom.training(
            x_train, y_oh_train, x_val_ps, y_oh_val_ps)
        _, test_sep = mod_sep.training(
            x_train_ps, y_oh_train_ps, x_val, y_oh_val)
        test_loss = [test_hom, test_sep]

    elif n_cv <= 2 and x_train_ps == None:
        mod = PhasePredictModel(
            model_type, model_params, phases, gpu_id=gpu_id)
        _, test_loss = mod.training(x_train, y_oh_train, x_val, y_oh_val)
        test_loss = [test_loss]
    else:
        mod = PhasePredictModel(
            model_type, model_params, phases, gpu_id=gpu_id)
        _, test_loss = mod.cross_val_train(x_train, y_oh_train, n_cv)

    score = -np.mean(test_loss)

    history.append(((space, score)))
    return {'loss': score, 'status': STATUS_OK}


def convert_parameters(history, params, pos=0):
    """Converts optimal parameters into a usable form by the model.
    Args:
        history (list[floats]): List of scores from optimization round
        params (dict): Model parameters
    Returns:
        optimized_params (dict): Parameters updated by optimized parameters
    """
    # Pull the desired position
    hyperopt_params = history[pos][0]
    optimized_params = params
    # Initiate keywords
    hyperopt_key = ['layer', 'dropout', 'activation', 'batchnorm']
    param_keys = ['layer_size', 'dropout', 'activation', 'batch_norm']
    last_layers = [2, 0, 'softmax', False]

    # Update parameters
    for opt_key, param_key, last_layer in zip(hyperopt_key, param_keys, last_layers):
        temp_params = [hyperopt_params[key]
                       for key in hyperopt_params.keys() if opt_key in key]
        if len(temp_params) > 0:
            optimized_params[param_key] = temp_params + [last_layer]
    return optimized_params


_DNN_architecture_space = {'num_layers'}

_DNN_hyperparam_space = {'lr'}

_DNN_regular_space = {'batch_size'}

_DNN_limited_hyperparam_space = {"activation_1", "activation_2", "activation_3",
                                 "batchnorm_1", "batchnorm_2", "batchnorm_3",
                                 "dropout_1", "dropout_2", "dropout_3", "lr"}

_XGB_hyperparam_space = {'num_parallel_tree', 'max_depth', 'colsample_bytree', 'subsample',
                         'alpha', 'gamma', 'learning_rate', 'lambda', 'num_boost_round',
                         'min_child_weight'}
_XGB_limited_hyperparam_space = {'max_depth', 'colsample_bytree', 'subsample',
                                 'alpha', 'gamma', 'learning_rate', 'lambda', 'min_child_weight'}

model_param_space = {'DNN': {'full': _DNN_hyperparam_space, 'architecture': _DNN_architecture_space,
                             'regularization': _DNN_regular_space},
                     'XGB': {'full': _XGB_hyperparam_space, 'secondary': _XGB_limited_hyperparam_space},
                     'GNN': {}}


def architecture_space_helper(n_layers):
    space = {
        '_'.join(['layer', str(i)]): hp.choice('_'.join(['layer', str(i), str(n_layers)]), [2**e for e in range(1, 8)])
        for i in range(1, n_layers)
    }

    space.update({
        '_'.join(['activation', str(i)]): hp.choice('_'.join(['activation', str(i), str(n_layers)]), ['LeakyReLU', 'relu'])
        for i in range(1, n_layers)
    })
    return space


def regularizer_space_helper(space, n_layers):
    space.update({
        '_'.join(['dropout', str(i)]): hp.uniform('_'.join(['dropout', str(i), str(n_layers)]), 0.0, 0.8)
        for i in range(1, n_layers)
    })

    space.update({
        '_'.join(['batchnorm', str(i)]): hp.choice('_'.join(['batchnorm', str(i), str(n_layers)]), [True, False])
        for i in range(1, n_layers)
    })
    return space


search_spaces = {'DNN': {'num_layers': hp.choice('num_layers', [architecture_space_helper(num_lay)
                                                                for num_lay in range(2, 9)]),
                         'lr': hp.uniform('lr', 0.001, 1),
                         'batch_size': hp.choice('batch_size', [2**e for e in range(5, 9)]),
                         'normalizer': hp.choice('normalizer', ['yj', 'minmax', 'mean'])},
                 'XGB': {'num_parallel_tree': hp.randint('num_parallel_tree', 2, 1000),
                         'max_depth': hp.randint('max_depth', 2, 12),
                         'min_child_weight': hp.uniform('min_child_weight', 0, 3),
                         'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
                         'subsample': hp.uniform('subsample', 0.2, 1),
                         'alpha': hp.uniform('alpha', 0, 10),
                         'lambda': hp.uniform('lambda', 0, 10),
                         'gamma': hp.uniform('gamma', 0, 5),
                         'learning_rate': hp.uniform('learning_rate', 0.01, 1),
                         'num_boost_round': hp.randint('num_boost_round', 2, 100)},
                 'GNN': {}}

transformers = {'yj': PowerTransformer(), 'minmax': MinMaxScaler(),
                'mean': StandardScaler()}
