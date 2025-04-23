from mamba_ssm.utils.generation import InferenceParams
import argparse

import torch
import os
import json
from custom_datasets.pg19 import *
import argparse
from task_deci import deci_pg19
from load_model_from_config import load_model, validate_config
from task_longbench import scorer, longbench_pred
from load_align_target import debug
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def my_deci(config, args=None):
    set_seed(config['seed'])
    merge_config = validate_config(config, args)

    # dataset settings
    path = './configs/deci_config.json'
    f = open(path)
    json_data = json.load(f)
    f.close()
    merge_config.update(json_data)
    merge_config["deci_dataset"] = args.deci_task

    model_processor, model, model_name = load_model(config, args)
    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "vanilla":
        model_name += "_vanilla"

    if args.deci_task in ["pg19", "yes"]:
        deci_pg19(model, model_processor, model_name, merge_config)

    
def my_longbench(config, args=None, only_eval=False):
    set_seed(config['seed'])
    merge_config = validate_config(config, args)  # define chunk size 1 or False

    model_processor, model, model_name = load_model(merge_config, args)
    if not only_eval:
        longbench_pred(args, model, model_processor, model_name, merge_config)

    scores = dict()
    if merge_config['model_arch'] == "ours":
        model_name += f"_{args.align_path}-{args.c}-{args.b}-{args.our_method}"
    elif merge_config['model_arch'] == "vanilla":
        model_name += "_vanilla"
    if args.long_eval_task == "e":
        path = f"pred_longbench_e/{model_name}/"
    else:
        path = f"pred_longbench/{model_name}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    if args.long_eval_task == "e":
        out_path = f"pred_longbench_e/{model_name}/result.json"
    else:
        out_path = f"pred_longbench/{model_name}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    

def special_input_ppl(config, args):
    import torch.nn.functional as F
    set_seed(config['seed'])
    merge_config = validate_config(config, args)

    tokenizer, model, model_name = load_model(config, args)

    with open(args.sample_path, 'r', encoding='utf-8') as file:
        inputs = file.read()
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

    ppls = []
    perplexities = {}
    max_amount_of_windows = 10
    length = [100e3, 80e3, 60e3, 40e3, 20e3, 10e3, 2e3]
    for window_size in length:
        window_size = int(window_size)
        nlls = []
        seq_len = inputs['input_ids'].size(1)
        if seq_len < window_size:
            print(f'skipping sample seq_len = {seq_len//1000}K < window_size = {window_size//1000}K')
            continue
        
        stride = (seq_len-window_size)//max_amount_of_windows
        if stride < 10:
            stride = 10

        for begin_loc in range(0, seq_len-window_size, stride):
            end_loc = begin_loc + window_size
            input_ids = inputs['input_ids'][:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()

            with torch.no_grad():
                inference_params = InferenceParams(max_seqlen=window_size+1, max_batch_size=input_ids.shape[0], merge_config=merge_config)
                target_ids = target_ids[:, -100:]
                if "Zamba2" in model_name:
                    outputs = model(input_ids, num_logits_to_keep=100+1, merge_config=merge_config)
                else:
                    outputs, _ = model(input_ids, num_last_tokens=100+1, inference_params=inference_params)
                logits = outputs.logits
                ce_loss = torch.nn.CrossEntropyLoss()
                neg_log_likelihood = ce_loss(logits.squeeze()[:-1], target_ids.squeeze())
                # print(neg_log_likelihood)
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).mean()).cpu().to(torch.float)
        print(f'{int(window_size/1e3)}k calculated perplexity: {ppl:.2f}')
        ppls.append(ppl)
        perplexities[f'{int(window_size/1e3)}k'] = f'{ppl:.4f}'
    avg_ppl = torch.stack(ppls).mean().item()
    perplexities["average"] = f'{avg_ppl:.4f}'
    os.makedirs(f'pred_ppl/{args.sample_path.split(".txt")[0]}', exist_ok=True)
    if args.model_arch == "ours":
        with open(f'pred_ppl/{args.sample_path.split(".txt")[0]}/{model_name}_{args.align_path}-{args.c}-{args.b}-{args.our_method}.json', "w") as f:
            json.dump(perplexities, f, ensure_ascii=False, indent=4)
    else:
        with open(f'pred_ppl/{args.sample_path.split(".txt")[0]}/{model_name}_{args.model_arch}.json', "w") as f:
            json.dump(perplexities, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    
    # Each new terminal's initial command ==> export HF_HOME='/research/data/zhifan/kxia_hf'
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default='0')
    parser.add_argument("--model", type=str, default="state-spaces/mamba2-1.3b")
    parser.add_argument("--model_arch", type=str, default="ours", choices=["vanilla", "ours"])

    # decay manipulation
    parser.add_argument("--align_path", type=str, default="longmamba")  # ./artifacts/ + model_name + align_path
    parser.add_argument("--our_method", type=str, default="dt_thre")  # alpha bound offline dt_thre norm
    
    # for special_input_ppl()
    parser.add_argument("--perplexity", "-ppl", action="store_true")
    parser.add_argument("--sample_path", type=str, default="subseq_lambada.txt")  # .txt dataset file path

    # DeciMamba PG19 tasks
    parser.add_argument("--deci_task", "-dt", type=str, default='no')    # choices=["no", "pg19", "yes"]
    # longBench tasks
    parser.add_argument("--long_eval_task", "-lt", type=str, default='no')    # choices=["no", "e", "yes"]
    parser.add_argument("--only_eval", action="store_true")  # only eval


    # hyperparameters
    parser.add_argument("--b", type=float, default=1.)  # alpha factor
    parser.add_argument("--c", type=float, default=5e-2)  # channel_threshold

    # parameter analysis
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_para4debug", "-p4d", action="store_true")  

    args = parser.parse_args()
    config = load_config(args)
    
    if args.debug:
        print("debug")
        debug(config, args)
        exit()
    if args.long_eval_task != "no":
        print("eval on long bench datasets")
        my_longbench(config, args, only_eval=args.only_eval)
    if args.deci_task != "no":
        print("eval on deci-mamba datasets")
        my_deci(config, args)
    if args.perplexity:
        special_input_ppl(config, args)

