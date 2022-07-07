import torch 
import numpy as np 


mask_dict = torch.load('deit_tiny_mask_init/deit_tiny_patch16_224_sparse0.5_after_train.pt')

new_mask5 = {}
new_mask20 = {}
new_mask50 = {}
new_mask70 = {}

indict_mask = {}

all_sequence = []
for key in mask_dict.keys():
    indict_mask[key] = torch.rand_like(mask_dict[key])
    all_sequence.append(indict_mask[key].reshape(-1))

all_sequence = torch.cat(all_sequence, 0)
nelement = all_sequence.shape[0]

sort_args = all_sequence.argsort()
index20 = sort_args[-int(nelement*0.2)]
index5 = sort_args[-int(nelement*0.05)]
index50 = sort_args[-int(nelement*0.5)]
index70 = sort_args[-int(nelement*0.7)]


for key in mask_dict.keys():
    ones = torch.ones_like(mask_dict[key])
    zeros = torch.zeros_like(mask_dict[key])
    new_mask5[key] = torch.where(indict_mask[key]<all_sequence[index5], zeros, ones)
    new_mask20[key] = torch.where(indict_mask[key]<all_sequence[index20], zeros, ones)
    new_mask50[key] = torch.where(indict_mask[key]<all_sequence[index50], zeros, ones)
    new_mask70[key] = torch.where(indict_mask[key]<all_sequence[index70], zeros, ones)

torch.save(new_mask5, 'deit_tiny_mask_init/rp_5.pt')
torch.save(new_mask20, 'deit_tiny_mask_init/rp_20.pt')
torch.save(new_mask50, 'deit_tiny_mask_init/rp_50.pt')
torch.save(new_mask70, 'deit_tiny_mask_init/rp_70.pt')
