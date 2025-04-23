import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from custom_datasets.pg19 import *
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(config, args):
    if "mamba" in config["base_model"]:
        wanted_dtype = torch.float32
        model_processor = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=True)

        if config['model_arch'] == 'vanilla' or 'ours' or 'deci':
            mamba_model_class = MambaLMHeadModel
        else:
            raise(f'bad mamba architecture: {config["model_arch"]}')
        
        if config['base_model'] is not None:
            print(f'loading model from checkpoint: {config["base_model"]}')
            model = mamba_model_class.from_pretrained(config['base_model'], device=f'cuda:{args.device}', dtype=wanted_dtype)
        else:
            print(f'no base_model set, loading model from checkpoint: state-spaces/mamba-1.4b')
            model = mamba_model_class.from_pretrained('state-spaces/mamba-1.4b', device=f'cuda:{args.device}', dtype=wanted_dtype)
    elif "Zamba2" in config["base_model"]:
        from zamba2.modeling_zamba2 import Zamba2ForCausalLM
        model = Zamba2ForCausalLM.from_pretrained(config['base_model'], torch_dtype=torch.bfloat16, revision="2269ae9c8a065c87dc1739ec99c9b81e5478082d").to(device=f'cuda:{args.device}')
        model_processor = AutoTokenizer.from_pretrained(config['base_model'], clean_up_tokenization_spaces=True)
    else:
        print(f'loading model from checkpoint: {config["base_model"]}')
        model = AutoModelForCausalLM.from_pretrained(config["base_model"], torch_dtype=torch.float32).to(f'cuda:{args.device}')
        model_processor = AutoTokenizer.from_pretrained(config["base_model"], clean_up_tokenization_spaces=True)

    model_name = config["base_model"].split("/")[-1]
    return model_processor, model, model_name


def get_decimation_config(config):
    decimation_config = {}
    decimation_config['activate'] = True
    decimation_config['record_debug_params'] = config['save_para4debug']

    decimation_config['beta'] = config['decimation_beta']
    decimation_config['min_seq_len'] = config['decimation_min_seq_len']  # 20
    decimation_config['type'] = config['decimation_type']  # max_p
    decimation_config['L_base'] = config['decimation_max_p_L_base']  # 2000
    decimation_config['decimating_layers'] = config['decimating_layers']  # 12
    
    return decimation_config


def validate_config(config, args):
    config["model_arch"] = args.model_arch
    config["base_model"] = args.model
    if "mba" not in config["base_model"]:
        print("Using non-Mamba model, reset 'model_arch' => 'vanilla'")
        config["model_arch"] = "vanilla"
    config["save_para4debug"] = args.save_para4debug
    config["b"] = args.b
    config["c"] = args.c
    config["align_path"] = config["base_model"].split("/")[-1] + "-" + args.align_path
    config["our_method"] = args.our_method
    print(config)

    if config['model_arch'] == 'deci':
        path = './configs/deci_config.json'
        f = open(path)
        json_data = json.load(f)
        f.close()
        # print(json_data)
        config.update(json_data)
        if "1.4b" in args.model:
            config["decimation_max_p_L_base"] = 4000
            config["decimation_beta"] = 1
            config["decimating_layers"] = [12]
        elif "2.8b" in args.model:
            config["decimation_max_p_L_base"] = 4000
            config["decimation_beta"] = 1
            config["decimating_layers"] = [22]
        if "mamba2" in args.model:
            config["decimation_max_p_L_base"] = 4000
            config["decimation_beta"] = args.beta
            config["decimating_layers"] = [6, 20, 22, 32]

    if config['model_arch'] == 'deci' and config['deci_dataset'] == 'ppl_test':
        config['deci_num_chunks'] = 2
    
    if not config['model_arch'] == 'deci':
        config['activate_decimation'] = False

    return config