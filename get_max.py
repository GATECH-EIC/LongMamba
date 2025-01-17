## compute max bound
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import os



root_path = "./artifacts"



for factor in ["0.00", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.10", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20"]:
    ref_list = [d for d in os.listdir(root_path) if f"1.2B-thepile_new4k-clampTop{factor}-0" in d]  # lambada0
    all_cnt = len(ref_list)
    print(all_cnt)
    for layer in range(38):
        max_alpha = {}
        max_decay = None
        max_tA_prod = None
        for dir in [d for d in os.listdir(root_path) if f"1.2B-thepile_new4k-clampTop{factor}-0" in d]:
            alpha_path = os.path.join(root_path, dir, "alpha", f"alpha_layer_{layer}.pt")
            decay_path = os.path.join(root_path, dir, "decay", f"decay_layer_{layer}.pt")
            tA_prod_path = os.path.join(root_path, dir, "tA_prod", f"tA_prod_layer_{layer}.pt")
            delta_thre_path = os.path.join(root_path, dir, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt")
            alpha = torch.load(alpha_path, map_location="cpu")
            decay = torch.load(decay_path, map_location="cpu")
            tA_prod = torch.load(tA_prod_path, map_location="cpu")
            delta_thre = torch.load(delta_thre_path, map_location="cpu")

            if max_decay is None:
                max_alpha = {key: value.clone() for key, value in alpha.items()}
                max_delta_thre = {key: value.clone() for key, value in delta_thre.items()}
                max_decay = decay.clone()
                max_tA_prod = tA_prod.clone()
            else:
                for key in alpha:
                    max_alpha[key] = torch.max(max_alpha[key], alpha[key])
                    max_delta_thre[key] = torch.max(max_delta_thre[key], delta_thre[key])
                max_decay = torch.max(max_decay, decay)
                max_tA_prod = torch.max(max_tA_prod, tA_prod)
        name = dir[:-2]+"max"
        os.makedirs(os.path.join(root_path, name, "alpha"), exist_ok=True)
        os.makedirs(os.path.join(root_path, name, "decay"), exist_ok=True)
        os.makedirs(os.path.join(root_path, name, "tA_prod"), exist_ok=True)
        os.makedirs(os.path.join(root_path, name, "delta_t-thre"), exist_ok=True)
        torch.save(max_alpha, os.path.join(root_path, name, "alpha", f"alpha_layer_{layer}.pt"))
        torch.save(max_decay, os.path.join(root_path, name, "decay", f"decay_layer_{layer}.pt"))
        torch.save(max_tA_prod, os.path.join(root_path, name, "tA_prod", f"tA_prod_layer_{layer}.pt"))
        torch.save(max_delta_thre, os.path.join(root_path, name, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt"))
        print(os.path.join(root_path, name, "delta_t-thre", f"delta_t-thre_layer_{layer}.pt"))