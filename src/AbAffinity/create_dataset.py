import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 

# Create dataset from sequences 
esm_alphabet = ['<cls>', '<pad>', '<eos>', '<unk>', 'L',  'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D',
 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

token2idx_dict = dict(zip(esm_alphabet, list(range(len(esm_alphabet)))))

def token2idx(token):
    if token in token2idx_dict:
        token = token2idx_dict[token]
    else:
        token = token2idx_dict['<unk>']
    return token

def idx2token(idx):
    for token, token_idx in token2idx_dict.items():
        if idx == token_idx:
            return token
    
def convert(seq, length=256):
    tokens = [token2idx('<cls>')] + [1] * length + [token2idx('<eos>')]
    if len(seq) > length:
        start = np.random.randint(len(seq)-length)
        seq = seq[start: start+length]
    for i, tok in enumerate(seq):
        tokens[i+1] = token2idx(tok)
    return np.array(tokens, dtype=int)


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, ab_seq, pred_affinity) -> None:
        super().__init__()
        self.ab_seq = ab_seq
        self.pred_affinity = pred_affinity 
        
    def __getitem__(self, idx):
        selected_ab = self.ab_seq[idx]
        affinity_val = self.pred_affinity[idx]
        return convert(selected_ab), float(affinity_val) 
    
    def __len__(self):
        return len(self.ab_seq) 


def get_dataloader(dataset, batch_size=8, num_workers=1):
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Set to True if you want to shuffle the data
        num_workers=num_workers,
        pin_memory=True  # Set to True if you have a GPU
    )

    return data_loader



def get_dataloaders(dataset, batch_size=8, seed=42, train_frac=0.8, num_workers=1):
    random_seed = seed 
    batch_size = batch_size 
    dataset_size = len(dataset)
    train_sample_num = int(dataset_size * train_frac)

    indices = list(range(dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices = indices[:train_sample_num]
    test_indices = indices[train_sample_num:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers,
        pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=test_sampler, 
        num_workers=num_workers,
        pin_memory=False)

    return train_loader, test_loader 

