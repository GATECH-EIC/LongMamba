import json
import numpy as np
import torch
import os
import random

     
def load_config(args):
    
    path = './configs/merge_config.json'

    f = open(path)
    json_data = json.load(f)
    f.close()

    if args.device not in ['None', "none", "no"]:
        json_data['model_device'] = f'cuda:{args.device}'

    return json_data

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
