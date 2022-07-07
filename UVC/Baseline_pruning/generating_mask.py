import os 
import copy 
import torch
import argparse
import numpy as np
import torch.nn as nn
from timm.models import create_model
from datasets import build_dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets.folder import ImageFolder

import models
from pruning_utils import *
from layers import Conv2d, Linear
from sp_vision_transformer import Attention, Mlp

def check_atten(binary_tensor, dim=64):
    heads_num = binary_tensor.shape[0] // dim
    output_list = []
    for i in range(heads_num):
        activated = binary_tensor[i*dim:i*dim+dim].mean().byte().float()
        output_list.append(activated.item())
    return output_list

def prune_loop(model, loss, pruner, dataloader, device, sparsity, scope, epochs, train_mode=False):

    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in range(epochs):
        pruner.score(model, loss, dataloader, device)
        sparse = sparsity**((epoch + 1) / epochs) # represent density actually.
        pruner.mask(sparse, scope)

def prune_conv_linear(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prune_conv_linear(model=module)

        if isinstance(module, nn.Linear):
            bias=True
            if module.bias == None:
                bias=False
            layer_new = Linear(module.in_features, module.out_features, bias)
            model._modules[name] = layer_new

        if isinstance(module, nn.Conv2d):
            layer_new = Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride)
            model._modules[name] = layer_new

    return model


parser = argparse.ArgumentParser('Pruning DeiT', add_help=False)
# Pruning parameters
parser.add_argument('--sparsity', default=0.5, type=float, help='density')

parser.add_argument('--atten_density', default=0.5, type=float, help='density')
parser.add_argument('--mlp_density', default=0.5, type=float, help='density')
parser.add_argument('--heads', default=12, type=int)

parser.add_argument('--pretrained', default=None, type=str, help='init or trained')
parser.add_argument('--save_file', default=None, type=str, help='save file name')
parser.add_argument('--data', default=None, type=str, help='save file name')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--type', default=None, type=str)


parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
args = parser.parse_args()

assert args.type

if args.type == 'synflow':

    print('#########################################')
    print('############ Synflow ####################')
    print('#########################################')

    number_examples = 100
    data = torch.ones(number_examples, 3, args.input_size, args.input_size)
    target = torch.ones(number_examples)
    data_set = torch.utils.data.TensorDataset(data, target)
    loader = torch.utils.data.DataLoader(data_set, 
        batch_size=1, shuffle=False, 
        num_workers=2, pin_memory=True)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    torch.save(model.state_dict(), 'random_init.pt')
    #pretrained or not
    if args.pretrained:
        print('loading pretrained weight')
        checkpoint = torch.hub.load_state_dict_from_url(
            args.pretrained, map_location='cpu', check_hash=True)
        model.load_state_dict(checkpoint['model'])

    save_state_dict = copy.deepcopy(model.state_dict())
    prune_conv_linear(model)

    for key in save_state_dict.keys():
        if not key in model.state_dict().keys():
            print('can not load key = {}'.format(key))

    model.load_state_dict(save_state_dict, strict=False)
    model.cuda()

    pruner = SynFlow(masked_parameters(model))
    prune_loop(model, None, pruner, loader, torch.device('cuda:0'), args.sparsity, scope='global', epochs=1, train_mode=True)
    print('Density = {}'.format(args.sparsity))

    current_mask = extract_mask(model.state_dict())
    check_sparsity_dict(current_mask)

    torch.save(current_mask, args.save_file)

elif args.type == 'mag':

    print('#########################################')
    print('########## Magnitude ####################')
    print('#########################################')

    number_examples = 100
    data = torch.ones(number_examples, 3, args.input_size, args.input_size)
    target = torch.ones(number_examples)
    data_set = torch.utils.data.TensorDataset(data, target)
    loader = torch.utils.data.DataLoader(data_set, 
        batch_size=1, shuffle=False, 
        num_workers=2, pin_memory=True)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print('loading pretrained weight')
    checkpoint = torch.hub.load_state_dict_from_url(
        args.pretrained, map_location='cpu', check_hash=True)
    model.load_state_dict(checkpoint['model'])

    save_state_dict = copy.deepcopy(model.state_dict())
    prune_conv_linear(model)

    for key in save_state_dict.keys():
        if not key in model.state_dict().keys():
            print('can not load key = {}'.format(key))

    model.load_state_dict(save_state_dict, strict=False)
    model.cuda()

    pruner = Mag(masked_parameters(model))
    prune_loop(model, None, pruner, loader, torch.device('cuda:0'), args.sparsity, scope='global', epochs=1, train_mode=True)
    print('Density = {}'.format(args.sparsity))

    current_mask = extract_mask(model.state_dict())
    check_sparsity_dict(current_mask)

    torch.save(current_mask, args.save_file)

elif args.type == 'taylor':
    
    print('##############################################')
    print('########### Taylor1Scorer ####################')
    print('##############################################')

    # prepare images for pruning  
    number_examples = 10000

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    all_dataset = datasets.ImageFolder(
        args.data,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    all_images = len(all_dataset)
    print('random pick {} images from dataset with {} images'.format(number_examples, all_images))
    select_index = np.random.permutation(all_images)[:number_examples]
    loader = torch.utils.data.DataLoader(
        Subset(all_dataset, list(select_index)), 
        batch_size=args.batch_size, shuffle=False, 
        num_workers=2, pin_memory=True)


    # prepare model for pruning 
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    print('loading pretrained weight')
    checkpoint = torch.hub.load_state_dict_from_url(
        args.pretrained, map_location='cpu', check_hash=True)
    model.load_state_dict(checkpoint['model'])

    save_state_dict = copy.deepcopy(model.state_dict())
    prune_conv_linear(model) # modify conv and linear layer with mask

    for key in save_state_dict.keys():
        if not key in model.state_dict().keys():
            print('can not load key = {}'.format(key))

    model.load_state_dict(save_state_dict, strict=False)
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    # start pruning 
    pruner = Taylor1ScorerAbs(masked_parameters(model))
    prune_loop(model, criterion, pruner, loader, torch.device('cuda:0'), args.sparsity, scope='global', epochs=1, train_mode=True)
    print('Density = {}'.format(args.sparsity))
    current_mask = extract_mask(model.state_dict())
    check_sparsity_dict(current_mask)
    torch.save(current_mask, args.save_file)

elif args.type == 'sp':


    print('##############################################')
    print('########### Sanity Pruning ###################')
    print('##############################################')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    all_dataset = datasets.ImageFolder(
        args.data,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    loader = torch.utils.data.DataLoader(all_dataset, 
        batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True)

    # prepare model for pruning 
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    print('loading pretrained weight')
    checkpoint = torch.hub.load_state_dict_from_url(
        args.pretrained, map_location='cpu', check_hash=True)
    model.load_state_dict(checkpoint['model'])

    save_state_dict = copy.deepcopy(model.state_dict())
    prune_conv_linear(model) # modify conv and linear layer with mask

    for key in save_state_dict.keys():
        if not key in model.state_dict().keys():
            print('can not load key = {}'.format(key))

    model.load_state_dict(save_state_dict, strict=False)
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()


    # start pruning 
    model.train()
    data, target = next(iter(loader))
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # attention 
    atten_mask = {}
    print('WARNING: single head dimention = 64')
    remain_heads = int(args.atten_density * args.heads)
    for name, m in model.named_modules():
        if isinstance(m, Attention):
            muti_head_dim = m.head_dim
            atten_dim = muti_head_dim * args.heads
            attn_mask_name = name + '.qkv.weight_mask'
            atten_mask[attn_mask_name] = torch.zeros_like(m.qkv.weight_mask)
            remain_indicator = torch.argsort(m.grad_scores)[-remain_heads:]

            for index_inside_layer in remain_indicator:
                q_start = index_inside_layer * muti_head_dim
                k_start = q_start + atten_dim
                v_start = k_start + atten_dim 
                atten_mask[attn_mask_name][q_start: q_start + muti_head_dim, :] = 1.
                atten_mask[attn_mask_name][k_start: k_start + muti_head_dim, :] = 1.
                atten_mask[attn_mask_name][v_start: v_start + muti_head_dim, :] = 1.

    #mlp 
    mlp_mask = {}
    for name, m in model.named_modules():
        if isinstance(m, Mlp):
            grad_fc1 = m.fc1.weight.grad 
            grad_fc2 = m.fc2.weight.grad 
            l1_norm = torch.norm(grad_fc1, dim=1, p=1) + torch.norm(grad_fc2, dim=0, p=1)
            remain_channel = int(args.mlp_density * l1_norm.shape[0])
            print(l1_norm.shape, remain_channel)

            #init mask 
            mask_name_fc1 = name + '.fc1.weight_mask'
            mask_name_fc2 = name + '.fc2.weight_mask'
            mlp_mask[mask_name_fc1] = torch.zeros_like(m.fc1.weight)
            mlp_mask[mask_name_fc2] = torch.zeros_like(m.fc2.weight)
            
            #enbale channel
            remain_indicator = torch.argsort(l1_norm)[-remain_channel:]
            mlp_mask[mask_name_fc1][remain_indicator, :] = 1.
            mlp_mask[mask_name_fc2][:,remain_indicator] = 1.

    model.load_state_dict(atten_mask, strict=False)
    model.load_state_dict(mlp_mask, strict=False)
    current_mask = extract_mask(model.state_dict())
    
    # show mask statistic
    heads_remain = 0
    heads_all = 0
    mlp_remain = 0
    mlp_all = 0

    for key in current_mask.keys():
        if '.qkv.weight_mask' in key:
            norm_mask = torch.norm(current_mask[key], dim=1, p=1)
            norm_mask = (norm_mask !=0).float()
            dim = norm_mask.shape[0] // 3
            q_act = check_atten(norm_mask[:dim])
            k_act = check_atten(norm_mask[dim:2*dim])
            v_act = check_atten(norm_mask[2*dim:])

            print(key, 'q', q_act)
            print(key, 'k', k_act)
            print(key, 'v', v_act)

        elif 'mlp.fc1.weight_mask' in key:
            norm_mask = torch.norm(current_mask[key], dim=1, p=1)
            norm_mask = (norm_mask != 0).float()
            nonzero = norm_mask.sum()
            nelement = norm_mask.shape[0]
            print(key, '{}/{}: {:.4f}'.format(nonzero, nelement, nonzero/nelement))

        elif 'mlp.fc2.weight_mask' in key:
            norm_mask = torch.norm(current_mask[key], dim=0, p=1)
            norm_mask = (norm_mask != 0).float()
            nonzero = norm_mask.sum()
            nelement = norm_mask.shape[0]
            print(key, '{}/{}: {:.4f}'.format(nonzero, nelement, nonzero/nelement))

    check_sparsity_dict(current_mask)

    torch.save(current_mask, args.save_file)





