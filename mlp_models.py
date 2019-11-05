import numpy as np
from sklearn.neural_network import MLPRegressor
np.random.seed(100)
def gen_model_parameters(num_models):
    neuronios =  [1,2,3,5,10,20,50,100]
    func_activation =  ['tanh','relu', 'identity']
    alg_treinamento = ['lbfgs','adam','sgd']
    learning_rate = ['constant','adaptive','invscaling']
    parameters = [neuronios,func_activation,alg_treinamento,learning_rate]
    selected_parameters = []
    for i in range(0,num_models):
        aux = []
        aux.append(np.random.choice(parameters[0]))
        aux.append(np.random.choice(parameters[1]))
        aux.append(np.random.choice(parameters[2]))
        aux.append(np.random.choice(parameters[3]))
        selected_parameters.append(aux)
    return selected_parameters

def gen_model(parameters):
    model = MLPRegressor(hidden_layer_sizes=parameters[0], activation=parameters[1], solver=parameters[2], max_iter = 10000, learning_rate=parameters[3])
    return model

def gen_all_models(num_models):
    sel_params = gen_model_parameters(num_models)
    all_models = []
    for parameters in sel_params:
        model = gen_model(parameters)
        all_models.append(model)
    return all_models


def n_best_models(models,indexes):
    best = []
    for i in indexes:
        best.append(models[i])
    return best