
import torch  
import numpy as np 
import matplotlib.pyplot as plt  
from .create_dataset import esm_alphabet, convert  
from .affinity_pred_model import AffinityPredictor
import textwrap
import random 
from collections import OrderedDict


device = 'cuda' if torch.cuda.is_available() else 'cpu'  

def AbAffinity():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AffinityPredictor().to(device) 
    state_dict = torch.load('model/2024-01-20_best_mse.pth',  map_location=device)
    # Remove the 'module.' prefix from keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.'
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = model.eval() 
    return model 

