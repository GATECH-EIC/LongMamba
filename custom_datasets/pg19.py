import torch
from torch.utils.data import Dataset, DataLoader

class PG19Dataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data
        
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        pg19_item = {}
        pg19_item['short_book_title'] = item['short_book_title']
        pg19_item['input_ids'] = item['input_tokens']
        return pg19_item

def get_pg19(val_only=False, model_name="mamba2"):
    
    if "mamba2" in model_name:
        mtype = "mamba"
    elif "Zamba2" in model_name:
        mtype = "zamba"
    val_set = torch.load(f'./artifacts/ppl_test/pg19/test_set_{mtype}.pt')
    dataset_val = PG19Dataset(val_set)

    if val_only:
        return dataset_val
    
    train_set = torch.load(f'./artifacts/ppl_test/pg19/train_set_{mtype}.pt')
    dataset_train = PG19Dataset(train_set)
    
    return dataset_train, dataset_val
