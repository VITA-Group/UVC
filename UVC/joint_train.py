# coding=utf-8
##
from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings('ignore')


import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
import time
import torch
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from models.model_distilled import DistilledVisionTransformer


from T2TViT.models.t2t_vit import *
from T2TViT.utils import load_for_transfer_learning



from functools import partial
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, PresetLRScheduler
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.losses import DistillationLoss
from uvc_optimizer import uvc_optimizer, uvc_optimizer_gating, build_minimax_model
import json
import copy
from uvc_utils import prune_w, prune_w_mask

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)
running_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

deit_family = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_tiny_patch16_224_half', 'deit_small_patch16_224_half', 'deit_base_patch16_224', 'deit_base_patch16_224_half',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224', 'deit_tiny_patch16_224_8layer',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384', 'deit_tiny_patch16_224_sp', 'deit_base_patch16_224_sp', 'deit_small_patch16_224_sp',
    'deit_base_distilled_patch16_384', 'deit_small_patch16_224_data'
]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_tau(max_tau, min_tau, ite, total):
    tau = min_tau + (max_tau - min_tau) * ite / total
    return tau


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def complex_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_model(args, model, minimax_model, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, args.name, f"{args.model_type}_{global_step}.pth.tar")
    save = {
        'args'      : args,
        'model'     : model_to_save.state_dict(),
        'stats_s'   : minimax_model.s if minimax_model is not None else None,
        'stats_r'   : minimax_model.r if minimax_model is not None else None
    }

    torch.save(model_to_save.state_dict(), model_checkpoint)
    if args.local_rank in [-1, 0]:
        print("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "imagenet":
        args.num_classes = 1000

    if "deit" in args.model_type:

        model = DistilledVisionTransformer(
            enable_dist=args.enable_deit,
            patch_size=config.patch_size, embed_dim=config.embed_dim, depth=config.depth, num_heads=config.num_heads, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0,
            gumbel_hard=False
        )


    elif "t2t" in args.model_type:
        # create model
        model = t2t_vit_14()

        # load the pretrained weights
        load_for_transfer_learning(model, args.model_path, use_ema=True, strict=False, num_classes=1000)

    else:
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes)
    # model.load_from(np.load(args.pretrained_dir))
    if args.pretrained == True:
        if args.model_path is not None: # 'need to specify teacher-path when using distillation'
            if args.local_rank in [-1, 0]:
                print(f"Loading checkpoint for model from ====> {args.model_path}")
            if args.model_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.model_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.model_path, map_location='cpu')

            try:
                model.load_state_dict(checkpoint["model"], strict=False)
            except Exception as e:
                # print(checkpoint["state_dict_ema"].keys())
                model.load_state_dict(checkpoint["state_dict_ema"], strict=False)

    model.to(args.device)
    for name, p in model.named_modules():
        if hasattr(p, "weight"):
            p.register_buffer("mask", torch.ones_like(p.weight))

    args.total_param = count_mask(model)

    # print("{}".format(config))
    if args.local_rank in [-1, 0]:
        print("Training parameters %s", args)
        print("Total Parameter: \t%2.1fM" % args.total_param)
    # print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def count_mask(model):
    total = 0
    for name, p in model.named_modules():
        if hasattr(p, "mask"):
            total += p.mask.sum()

    return total / 1e6


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    top1        = AverageMeter()
    if args.local_rank in [-1, 0]:
        print("***** Running Validation *****")
        print("  Num steps = %d", len(test_loader))
        print("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... [loss=X.X]",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True, disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        if args.enable_patch_gating==2:
            tau = 1
        else:
            tau = -1

        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            if args.distillation_type == None:
                logits = model(x)[0]
            else:
                logits, flops_list = model(x ,tau, args.patch_ratio)
                # logits = model(x)

            eval_loss = loss_fct(logits, y)
            prec1 = complex_accuracy(logits.data, y)[0]
            top1.update(prec1.item(), x.size(0))
            eval_losses.update(eval_loss.item())

        epoch_iterator.set_description("Validating... [loss=%2.5f] | [Acc=%2.5f]" %
            (eval_losses.val, top1.val) )
        # break

    if args.local_rank in [-1, 0]:
        print("\n")
        print("Validation Results")
        print("Global Steps: %d" % global_step)
        print("Valid Loss: %2.5f" % eval_losses.avg)
        print("Valid Accuracy: %2.5f" % top1.avg)
    # print("Valid Flops: %2.5f" % )
    if args.enable_writer :
        writer.add_scalar("test/accuracy", scalar_value=top1.avg, global_step=global_step)
    return top1.avg


def train(args, model, uvc_args=None, mixup_fn=None, criterion=None):
    """ Train the model """
    writer = None
    if args.enable_pruning:
        uvc_optimizer_func = uvc_optimizer
    else:
        uvc_optimizer_func = uvc_optimizer_gating

    if args.enable_writer and args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir,args.name), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(
            "uvc_train",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) [:10]
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # t_total = args.num_steps
    t_total = len(train_loader) * args.num_epochs # modify
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    zlr_scheduler = PresetLRScheduler(args.zlr_schedule)



    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank!=-1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size(), delay_allreduce=True)
    else:
        model = nn.DataParallel(model)
    # Train!
    if args.local_rank in [-1, 0]:
        print("***** [Stage 1] Training with ADMM *****")
        print(f"  Total optimization steps = {args.num_steps}", )
        print(f"  Instantaneous batch size per GPU = {args.train_batch_size}", )
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    # global_step, best_acc = 0, 0
    global_step = 0
    best_acc = 0
    # import json

    if not os.path.exists(os.path.join(args.output_dir, args.name)):
        os.makedirs(os.path.join(args.output_dir, args.name))
    init = json.dumps({})
    f = open(os.path.join(args.output_dir, args.name, f"s_{running_time}.json"),"w")
    f.write(init)
    f.close()
    f = open(os.path.join(args.output_dir, args.name, f"r_{running_time}.json"), "w")
    f.write(init)
    f.close()
    f = open(os.path.join(args.output_dir, args.name, f"gating_{running_time}.json"), "w")
    f.write(init)
    f.close()


    # print("Validation before UVC training")
    # accuracy = valid(args, teacher, writer, test_loader, global_step)

    delta1_log = []
    delta2_log = []
    stage = "UVC Train"

    if True:
        model.train()
        epoch = 0
        total_epochs = args.num_epochs
        while epoch <= args.num_epochs:
            epoch += 1
            gating_grad_list = []

            ## Monitoring Sparsity of each epoch
            minimax_model, dual_optimizer, s_optimizer, r_optimizer, gating_optimizer = uvc_args


            # Warmup for fixed skip gating
            if epoch<=args.warmup_epochs:
                stage = "Warm Up"
                total_epochs = args.warmup_epochs
                args.gumbel_hard   = 1
                minimax_model.model.enable_warmup = 1
                minimax_model.model.block_skip_gating.requires_grad = False
                for params in optimizer.param_groups:
                    params['lr'] = args.warmup_lr


            if epoch>args.warmup_epochs:
                total_epochs = args.num_epochs
                stage = "UVC Train"
                minimax_model.model.enable_warmup = 0
                args.enable_warmup = 0
                args.gumbel_hard   = 0
                minimax_model.model.block_skip_gating.requires_grad = True

                if epoch == args.warmup_epochs+1 and args.warmup_reset:
                    if args.local_rank in [-1, 0]:
                        print(" Reset the Optimizer and Learning rate scheduler")
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                    t_total = len(train_loader) * args.num_epochs # modify
                    if args.decay_type == "cosine":
                        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
                    else:
                        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)






            prune_w_mask(minimax_model, optimizer)
            remained_param = count_mask(minimax_model.model)
            if args.local_rank in [-1, 0]:
                print("="*60)
                print(f"Start [Epoch {epoch}] at Stage {stage}")
                print(f"[Initial Sparsity|Epoch {epoch}] Parameter size: {(remained_param):.2f}M / {args.total_param:.2f}M = {(remained_param)/args.total_param*100:.2f}%")

            # Decay for epsilon of softl0
            if stage == "UVC Train":
                minimax_model.update_eps()

            epoch_iterator = tqdm(train_loader,
                                  desc="Stage [X / X Epochs] [X / X Steps] [LR: X | Loss: X.XXX]",
                                  # desc="Training [X / X Steps] [loss = X.X]",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)

            start_time =  time.time()
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch

                if len(x) % 2 != 0:
                    x = x[:-1]
                    y = y[:-1]


                if args.enable_patch_gating == 2:
                    tau = get_tau(10, 0.1, global_step, args.num_epochs*len(train_loader))
                else:
                    tau = -1

                x, y = mixup_fn(x, y)
                outputs, flops_list = model(x, tau, args.patch_ratio)
                loss = criterion(x, outputs, y)


                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                if (step + 1) % args.gradient_accumulation_steps == 0:
                    losses.update(loss.item()*args.gradient_accumulation_steps)
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()


                    global_step += 1
                    if args.uvc_train:
                        infos = {"global_step": global_step}
                        save_budgets = sorted([float(i) for i in args.save_budgets.split(',')])
                        minimax_model, dual_optimizer, s_optimizer, r_optimizer, gating_optimizer = uvc_args


                        # Increase zlr for every 5 epochs
                        if not minimax_model.model.enable_warmup:
                            zlr_scheduler(dual_optimizer, epoch, "zlr")
                        minimax_model.update_gating()
                        cur_resource, s_data, r_data, gating_data, gating_grad_list = uvc_optimizer_func(optimizer, minimax_model, s_optimizer, r_optimizer, gating_optimizer, dual_optimizer, args, infos, save_budgets, flops_list, args.z_grad_clip, global_step, args.gating_interval, gating_grad_list)



                    optimizer.zero_grad()

                    epoch_iterator.set_description(
                        f"{stage} [{epoch} / {total_epochs} Epochs] [{global_step} / {t_total} Steps] [LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {losses.val:.3f}]"
                    )
                    if args.local_rank in [-1, 0]:
                        if args.enable_writer:
                            writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
                        if args.uvc_train:
                            if args.enable_writer:
                                writer.add_scalar("resource",scalar_value=cur_resource,global_step=global_step)
                                writer.add_scalar("s_sample", scalar_value=s_data[0][0], global_step=global_step)
                                writer.add_scalar("r_sample", scalar_value=r_data[0][0], global_step=global_step)
                            if global_step % args.log_interval == 0 and epoch>args.warmup_epochs:

                                s_dict = {global_step: s_data.tolist()}
                                r_dict = {global_step: r_data.tolist()}
                                with open(os.path.join(args.output_dir,args.name,f"s_{running_time}.json"),"r+") as file:
                                    data = json.load(file)
                                    data.update(s_dict)
                                    file.seek(0)
                                    json.dump(data,file)
                                with open(os.path.join(args.output_dir,args.name,f"r_{running_time}.json"),"r+") as file:
                                    data = json.load(file)
                                    data.update(r_dict)
                                    file.seek(0)
                                    json.dump(data,file)


                                if args.enable_block_gating:
                                    gating_dict = {global_step: gating_data.tolist()}
                                    with open(os.path.join(args.output_dir,args.name,f"gating_{running_time}.json"),"r+") as file:
                                        data = json.load(file)
                                        data.update(gating_dict)
                                        file.seek(0)
                                        json.dump(data,file)

                    start_time = time.time()




            if args.local_rank in [-1, 0]:
                print()

                print("*"*60)
                print("Epoch finished, begin validating ...")
            accuracy = valid(args, model, writer, test_loader, global_step)

            prune_w_mask(minimax_model, optimizer)
            remained_param = count_mask(minimax_model.model)
            save_model(args, minimax_model.model, minimax_model, epoch)

            # print(minimax_model.s)
            # print(minimax_model.r)
            # print(minimax_model.block_skip_gating)
            if args.local_rank in [-1, 0]:
                print()
                print(f"[Validation Sparsity|Step {global_step}|Epoch {epoch}]")
                print(f"Parameter size: {(remained_param):.2f}M / {args.total_param:.2f}M = {(remained_param)/args.total_param*100:.2f}%")
                print(f"Expectation FLOPs: {minimax_model.run_resource_fn(args.gumbel_hard)*100}%", f"Real FLOPs: {minimax_model.run_resource_fn(gumbel_hard=True)*100}%")


            if args.enable_writer and args.local_rank in [-1, 0]:
                writer.add_scalar("train/param_size", scalar_value=(remained_param)/args.total_param, global_step=global_step)
                writer.add_scalar("train/flops_size", scalar_value=minimax_model.run_resource_fn(args.gumbel_hard)*100, global_step=global_step)
            if args.local_rank in [-1, 0]:
                print("*"*60)
                print()

            model.train()
            losses.reset()
            # if global_step % t_total == 0:
            #     break


    if args.local_rank in [-1, 0] and args.enable_writer:
        writer.close()
    print("Best Accuracy: \t%f" % best_acc)
    print("End Training!")

    return minimax_model

def get_uvc_layers(model, args=None):
    layer_names = {}
    layer_names[None] = None
    uvc_layers = {"W1":[], "W2":[],"W3":[]}

    for name, m in model.named_modules():
        if hasattr(m, "in_features"): # Block out dropouts
            if "mlp.fc2" in name or "attn.proj" in name:
                layer_names[m] = name
                if "attn.proj" in name:
                    uvc_layers["W1"].append(m)
                    m.uvc_s = 0
                else:
                    uvc_layers["W3"].append(m)
                    m.uvc_s = 0

            if "mlp.fc1" in name:
                # print('name',name)
                # print("m",m)
                layer_names[m] = name
                uvc_layers["W2"].append(m)

                m.uvc_s = 0



    uvc_layers_dict = {"s_dict":{},"r_dict":{}}

    for i,m in enumerate(uvc_layers["W1"]):
        uvc_layers_dict["s_dict"][m] = [i,0]
        uvc_layers_dict["r_dict"][m] = i
    for i,m in enumerate(uvc_layers["W3"]):
        uvc_layers_dict["s_dict"][m] = [i,1]
    print('=================================================')
    return layer_names, uvc_layers, uvc_layers_dict


def post_training(args, model, mixup_fn=None, criterion=None):

    print("Starting post training")
    train_loader, test_loader = get_loader(args)
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.post_learning_rate, weight_decay=args.post_weight_decay)

    # t_total = args.num_steps
    t_total = len(train_loader) * args.num_epochs # modify
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20




    if args.local_rank!=-1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    else:
        model = nn.DataParallel(model)



    # Train!
    if args.local_rank in [-1, 0]:
        print("***** [Stage 2] Post Training *****")
        print("  Instantaneous batch size per GPU = %d", args.train_batch_size)


    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    # global_step, best_acc = 0, 0
    global_step = 0
    best_acc = 0

    delta1_log = []
    delta2_log = []
    model.train()


    for epoch in range(args.num_epochs):
        if args.local_rank in [-1, 0]:
            print("="*60)
            print(f"Start training [Epoch {epoch}]")


        epoch_iterator = tqdm(train_loader,
                      desc="Training [X / X Steps] [LR: X | Loss: X.XXX]",
                      # desc="Training [X / X Steps] [loss = X.X]",
                      bar_format="{l_bar}{r_bar}",
                      dynamic_ncols=True,
                      disable=args.local_rank not in [-1, 0])
        start_time = time.time()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            if len(x) % 2 != 0:
                x = x[:-1]
                y = y[:-1]
            for name, m in model.named_modules():
                if hasattr(m, "mask"):
                    # print(name, m)
                    m.weight.data *= m.mask

            # print(f"Data time: {time.time() - start_time}")
            x, y = mixup_fn(x, y)
            outputs, flops_list = model(x)
            loss = criterion(x, outputs, y)
            # print(f"Forward time: {time.time() - start_time}")
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # print(f"Backward time: {time.time() - start_time}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()

                global_step += 1
                optimizer.zero_grad()


                epoch_iterator.set_description(
                    "Training [%d / %d Steps] [LR: %.6f | Loss: %.3f]" % (global_step, t_total, scheduler.get_last_lr()[0], losses.val)
                )

                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)


                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)

                    if best_acc < accuracy:
                        save_model(args, model, None, global_step)
                        best_acc = accuracy
                    model.train()
            start_time = time.time()




def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default="debug",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10","cifar100","imagenet"], default="imagenet",
                        help="Which downstream task.")
    # parser.add_argument("--data_dir", default="/home/shixing/dataset")
    parser.add_argument("--data_dir", default="/ssd1/shixing/imagenet2012",
                        help="directory of data")

    parser.add_argument("--num_workers", default=4, type = int)
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"] + deit_family + ["t2t_vit_14"],
                        default="deit_tiny_distilled_patch16_224",
                        help="Which variant to use.")
    parser.add_argument("--model_path", default=None,
                        help="State dict path")
    parser.add_argument("--pretrained_dir", type=str, default="../ViT-pytorch/pretrain/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained", type=int, default=1,
                        help="Whether we use a pretrained ViT models.")
    parser.add_argument("--output_dir", default="../result/output/uvc_train", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=1024, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")


    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--num_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    ### new args from uvc training
    parser.add_argument('--uvc_train', action = 'store_true', default=True,
                        help="if performing unified compression or not")
    parser.add_argument('--soptim', default='sgd',
                        help='optimizer for DNN Sparsity training')
    parser.add_argument('--roptim', default='sgd',
                        help='optimizer for DNN Sparsity training')
    parser.add_argument('--zlr_schedule_list', default="10,20,30,40,50", type=str,
                        help='dual lr for z')

    parser.add_argument('--ylr', default=1e-4, type=float,
                        help='dual lr for y')
    parser.add_argument('--plr', default=1e-4, type=float,
                        help='dual lr for p')
    parser.add_argument('--slr', default=0.02, type=float,
                        help='primal lr for s')
    parser.add_argument('--rlr', default=0.02, type=float,
                        help='primal lr for r')
    parser.add_argument('--glr', default=1e-3, type=float,
                        help='primal lr for gating parameters')
    parser.add_argument('--log_interval', default=2000, type=int,
                        help='time interval for printing compression details')
    parser.add_argument('--save_budgets',
                        default='0.6, 0.5, 0.4',
                        help='budgets to save checkpoints')
    parser.add_argument('--budget',
                        default=0.5,
                        help='budget of model compression')
    parser.add_argument('--sl2wd', default=0.0, type=float,
                        help='l2 weight decay for s')
    parser.add_argument('--verbose', default=True,action='store_true',
                        help='set verbose to be true to print uvc infos')




    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.8,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


    # Distillation parameters
    parser.add_argument('--teacher-model', default=None, type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default=None)
    parser.add_argument('--distillation-type', default='hard', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')



    ## Argument for post training
    parser.add_argument("--post_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--post_weight_decay", default=0.05, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--post_num_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")



    parser.add_argument("--use_distribute", default=1, type=int,
                        help="Whether launch distributed learning")
    parser.add_argument("--enable_writer", default=0, type=int,
                        help="Using or not using tensorboard writer")
    parser.add_argument("--flops_with_mhsa", type=int, default=1,
                        help="Whether calculate flops with Multi-Head Self Attention")
    parser.add_argument("--enable_block_gating", type=int, default=1,
                        help="Whether pass through block")
    parser.add_argument("--enable_part_gating", type=int, default=0,
                        help="Whether pass through attention and mlp")
    parser.add_argument("--enable_jumping", type=int, default=0,
                        help="Whether jump the past connection to the last layer")
    parser.add_argument("--enable_deit", type=int, default=0,
                        help="whether use deit structure")
    parser.add_argument("--enable_pruning", type=int, default=1,
                        help="whether use pruning within a block")
    parser.add_argument("--enable_patch_gating", type=int, default=0,
                        help="whether use patch slimming to reduce the dimension of patch length")
    parser.add_argument("--patch_ratio", type=float, default=0.9,
                        help="The ratio of patch chosen")


    parser.add_argument('--z_grad_clip', default=0.5, type=float,
                        help='Gradident clipping parameter of z')
    parser.add_argument('--gating_interval', default=100, type=int,
                        help='The update interval of gating variable')
    parser.add_argument('--gating_weight', default=5, type=float,
                        help='block gating gradient weight of resource function')
    parser.add_argument('--patch_weight', default=5, type=float,
                        help='Patch Gating gradient weight of resource function')
    parser.add_argument('--patch_l1_weight', default=0.01, type=float,
                        help='The l1 loss weight of patch gating')
    parser.add_argument('--patchlr', default=0.01, type=float,
                        help='Learning rate of patch gating')
    parser.add_argument('--patchloss', default="l1", type=str,
                        help='Learning rate of patch gating')
    parser.add_argument('--use_gumbel', default=1, type=int,
                        help='Learning rate of patch gating')
    parser.add_argument('--eps', default=0.1, type=float,
                        help='The epsilon if we use softl0 as the function to select gating path')
    parser.add_argument('--eps_decay', default=0.92, type=float,
                        help='The decay rate of epsilon for each epoch')

    parser.add_argument('--enable_warmup', default=1, type=int,
                        help='Warmup for skip connection')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='Warmup epoch numbers')
    parser.add_argument('--warmup_lr', default=1e-4, type=float,
                        help='Warmup learning')
    parser.add_argument('--warmup_reset', default=0, type=int,
                        help='Whether reset optimizer and scheduler after warm up')





    parser.add_argument("--gpu_num", type=str, default="0, 1",
                        help="Used gpu number")

    args = parser.parse_args()

    args.budget = float(args.budget)

    config = CONFIGS[args.model_type]
    args.head_size = config.hidden_size // config.transformer["num_heads"]
    args.num_heads = config.transformer["num_heads"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    # Setup CUDA, GPU & distributed training
    if args.local_rank==-1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, args.use_distribute, args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)


    teacher_model = None

    # Set up mix-up function

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if args.local_rank in [-1, 0]:
        print(f"mixup active: {mixup_active}")

    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)


    # Set up criterion

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Set up teacher model

    if args.distillation_type!='none':
        if args.teacher_model == None:
            args.teacher_model = args.model_type
        if args.teacher_path == None:
            args.teacher_path  = args.model_path
        if args.local_rank in [-1, 0]:
            print(f"** Creating teacher model: [{args.teacher_model}] loading from ==> [{args.teacher_path}]")

        if 'deit' in args.teacher_model:
            teacher_model = DistilledVisionTransformer(
                enable_dist=args.enable_deit,
                patch_size=config.patch_size, embed_dim=config.embed_dim, depth=config.depth, num_heads=config.num_heads, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0,
            )

        elif "t2t" in args.teacher_model:
            teacher_model = t2t_vit_14(pretrained=True)
            if args.local_rank in [-1, 0]:
                print(f"-----Has teacher_model [{args.teacher_model}]-----")


        if args.teacher_path is not None: # 'need to specify teacher-path when using distillation'
            if args.teacher_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.teacher_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.teacher_path, map_location='cpu')
            try:
                teacher_model.load_state_dict(checkpoint["model"], strict=False)
            except Exception as e:
                teacher_model.load_state_dict(checkpoint["state_dict_ema"], strict=False)

        teacher_model.to(device)
        teacher_model.eval()

    train_loader, test_loader = get_loader(args)

    # Construct distillation loss
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    if args.distillation_type == 'none':
        args.distillation_type = None





    if args.uvc_train:


        args.zlr_schedule_list = args.zlr_schedule_list.split(",")
        zlr_epoch_gap = args.num_epochs // len(args.zlr_schedule_list)
        zlr_schedule = {}
        for i in range(len(args.zlr_schedule_list)):
            args.zlr_schedule_list[i]       = int(args.zlr_schedule_list[i])
            zlr_schedule[i*zlr_epoch_gap]   = int(args.zlr_schedule_list[i])
        args.zlr_schedule = zlr_schedule

        layer_names, uvc_layers, uvc_layers_dict = get_uvc_layers(model, args) # modify 1


        with torch.no_grad():
            model.eval()
            y, flops_list = model(torch.ones(1, 3, 224, 224).cuda(), number=args.patch_ratio)

        minimax_model, dual_optimizer, s_optimizer,r_optimizer, gating_optimizer = build_minimax_model(model, layer_names, uvc_layers,
                                                                         uvc_layers_dict, args, flops_list) # modify 2
        minimax_model = minimax_model.cuda()
        uvc_args = [minimax_model, dual_optimizer, s_optimizer, r_optimizer, gating_optimizer]

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

        infos = {"step", 0}
        save_budgets = sorted([float(i) for i in args.save_budgets.split(',')])
        prune_w_mask(minimax_model,optimizer)

        minimax_model = train(args, model, uvc_args=uvc_args, mixup_fn = mixup_fn, criterion=criterion)



        prune_w_mask(minimax_model,optimizer)
        post_training(args, minimax_model.model, mixup_fn, criterion)


    else:
        train(args, model)

    # Training




if __name__ == "__main__":
    main()
