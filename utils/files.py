import os 
import numpy as np

import time 

import config
import torch

def load_model(env, name, exploiter=False):
    if exploiter:
        filename = os.path.join(config.EXPMODELDIR, name)
    else:
        filename = os.path.join(config.MODELDIR, name)
    if os.path.exists(filename):
        sucess = False 
        #carga si falla iterativa por si el exploiter o el main justo meten un modelo nuevo
        while not sucess:
            try:
                print(f'Loading {name}')
                model = torch.jit.load(filename).to('cpu')
                sucess = True 
            except Exception as e:
                print(f"Error loading model! Retrying... Error: {e}")
                sucess = False
                time.sleep(1)         
    else:
        raise Exception(f'\n{filename} not found')
    
    model.eval()
    return model


def load_all_models(env):
    modellist = [f for f in os.listdir(config.MODELDIR) if f.startswith("_model")]
    modellist.sort()
    models = [load_model(env, "base.ts")]
    model_filenames = ["base.ts"]
    for model_name in modellist[-config.WINDOW_MODELS:]:
        models.append(load_model(env, model_name))
        model_filenames.append(model_name)
    return models, model_filenames


def load_best_model(env):
    filename = get_best_model_name()

    model = load_model(env, filename)

    return model, filename


def get_best_model_name(exploiter=False):
    if exploiter:
        #los exploiters no pelean contra si mismos
        modellist = [f for f in os.listdir(os.path.join(config.EXPMODELDIR)) if f.startswith("_model")]

    else:
        modellist = [f for f in os.listdir(os.path.join(config.MODELDIR)) if f.startswith("_model")]

    #modellist = [f for f in os.listdir(os.path.join(config.MODELDIR)) if f.startswith("_model")]

    if len(modellist)==0:
        if exploiter:
            filename = None 
        else:
            filename = 'base.ts'
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename


def get_model_stats(filename):
    if filename == 'base.ts':
        generation = 0
        timesteps = 0
        best_reward = -np.inf
    else:
        stats = filename.split('_')
        generation = int(stats[2])
        best_reward = float(stats[3])
        timesteps = int(stats[4])
    return generation, timesteps, best_reward