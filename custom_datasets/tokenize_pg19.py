import torch
from tqdm import tqdm
from datasets import load_dataset
import argparse
from transformers import AutoTokenizer
from itertools import islice
model_processor_mamba = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model_processor_zamba = AutoTokenizer.from_pretrained("Zyphra/Zamba2-1.2B")

def main(tokenizer_type="mamba"):
    splits = ['test']
    if tokenizer_type == "mamba":
        model_processor = model_processor_mamba
    elif tokenizer_type == "zamba":
        model_processor = model_processor_zamba
    
    for split in splits:
        print(f'Tokenizing {split} split')
        streaming_dataset = load_dataset("deepmind/pg19", cache_dir='./hf_cache', streaming=True)
        cur_dataset = list(islice(streaming_dataset, 3001))
        i=0
        tokenized_ds = []
        for sample in tqdm(cur_dataset):
            cur_sample = {}
            cur_sample['short_book_title'] = sample['short_book_title']
            cur_sample['input_tokens'] = model_processor(text=sample['text'], return_tensors="pt").input_ids
            tokenized_ds.append(cur_sample)

            i+=1
            if i%1000 == 0 and i>0:
                print(f'saving checkpoint after {i} examples')
                torch.save(tokenized_ds, f'./artifacts/ppl_test/{split}_set_{tokenizer_type}.pt')
            if i>2001:
                break
        print("tokenized_ds: ", len(tokenized_ds))
        torch.save(tokenized_ds, f'./artifacts/ppl_test/{split}_set_{tokenizer_type}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main("mamba")