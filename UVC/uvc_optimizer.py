import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

from uvc_utils import UVC_CP_MiniMax, \
                    flops2 , \
                    calc_flops, \
                     prox_w, \
                     proj_dual,\
                     array1d_repr, array2d_repr, \
                     weight_list_to_scores

class SoftL0(torch.nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, inputs):
        loss = torch.sqrt(inputs**2 / (inputs**2 + self.eps))
        return loss



def uvc_optimizer(optimizer, minimax_model, s_optimizer, r_optimizer, gating_optimizer, dual_optimizer, args, infos, save_budgets, flops_list, z_grad_clip, global_step, gating_interval, gating_grad_list):
    s_max = (minimax_model.s_ub - 1 - 1e-8).clamp(min=0.0)
    r_max = (minimax_model.r_ub - 1 - 1e-8).clamp(min=0.0)


    prox_w(minimax_model, optimizer)
    # print(minimax_model.block_skip_gating)


    s_loss1 = minimax_model.sloss1()
    r_loss1 = minimax_model.rloss1()
    sr_loss2 = minimax_model.srloss2(budget = args.budget)
    cur_resource = sr_loss2.item() + args.budget
    sr_loss2 = sr_loss2.clamp(-z_grad_clip, z_grad_clip)

    if minimax_model.model.enable_warmup:
        if minimax_model.block_skip_gating is not None:
            out_gating = minimax_model.block_skip_gating.cpu().data.numpy()
        else:
            out_gating = None

        return cur_resource, minimax_model.s.cpu().data.numpy(), minimax_model.r.cpu().data.numpy(), out_gating,gating_grad_list

    # resource_ub =  7.07973120e+07
    resource_ub = 1

    s_grad1 = torch.autograd.grad(s_loss1, minimax_model.s, only_inputs=True)[0].data \
              + args.sl2wd * (minimax_model.s.data / minimax_model.s_ub)  # >=0
    # print("s_grad1", s_grad1)
    r_grad1 = torch.autograd.grad(r_loss1, minimax_model.r , only_inputs=True)[0].data + args.sl2wd * (minimax_model.r.data / minimax_model.r_ub)  # >=0



    with torch.autograd.set_detect_anomaly(True):
        if args.enable_patch_gating==1 and args.enable_block_gating:
            s_grad2_temp, r_grad2_temp, g_grad_resource, patch_grad_resource = torch.autograd.grad(sr_loss2, [minimax_model.s,minimax_model.r, minimax_model.block_skip_gating, minimax_model.model.patch_gating], only_inputs=True,allow_unused=True)
        elif args.enable_block_gating:
            s_grad2_temp, r_grad2_temp, g_grad_resource = torch.autograd.grad(sr_loss2, [minimax_model.s,minimax_model.r, minimax_model.block_skip_gating], only_inputs=True,allow_unused=True)
        elif args.enable_patch_gating==1:
            s_grad2_temp, r_grad2_temp, patch_grad_resource = torch.autograd.grad(sr_loss2, [minimax_model.s,minimax_model.r, minimax_model.model.patch_gating], only_inputs=True,allow_unused=True)
        else:
            s_grad2_temp, r_grad2_temp = torch.autograd.grad(sr_loss2, [minimax_model.s,minimax_model.r], only_inputs=True,allow_unused=True)

    s_grad2 = s_grad2_temp.data * resource_ub
    r_grad2 = r_grad2_temp.data * resource_ub

    s_optimizer.zero_grad()
    minimax_model.s.grad = s_grad1 + minimax_model.z.data * s_grad2

    r_optimizer.zero_grad()
    minimax_model.r.grad = r_grad1 + minimax_model.z.data * r_grad2

    if gating_optimizer is not None:
        g_grad = minimax_model.block_skip_gating.grad + minimax_model.z.data * args.gating_weight * g_grad_resource
        gating_grad_list.append(g_grad.unsqueeze(0)*(global_step%gating_interval) )
        gating_optimizer.zero_grad()

        if (global_step+1) % gating_interval == 0:
            minimax_model.block_skip_gating.grad = torch.cat(gating_grad_list).mean(0)
            gating_optimizer.step()
            gating_grad_list = []
            gating_optimizer.zero_grad()

    overflow_idx = minimax_model.s.data >= s_max
    underflow_idx = minimax_model.s.data <= 0


    minimax_model.s.grad.data[overflow_idx] = minimax_model.s.grad.data[overflow_idx].clamp(min=0.0)
    minimax_model.s.grad.data[underflow_idx] = minimax_model.s.grad.data[underflow_idx].clamp(max=0.0)

    clip_grad_norm_(minimax_model.s, 1.0, float('inf'))
    # print("minimax_model.s.grad_final",minimax_model.s.grad)
    s_optimizer.step()
    minimax_model.s.data.clamp_(min=0.0)
    minimax_model.s.data[overflow_idx] = s_max[overflow_idx]

    # --
    overflow_idx = minimax_model.r.data >= r_max
    underflow_idx = minimax_model.r.data <= 0

    minimax_model.r.grad.data[overflow_idx] = minimax_model.r.grad.data[overflow_idx].clamp(min=0.0)
    minimax_model.r.grad.data[underflow_idx] = minimax_model.r.grad.data[underflow_idx].clamp(max=0.0)

    clip_grad_norm_(minimax_model.r, 1.0, float('inf'))
    r_optimizer.step()

    minimax_model.r.data.clamp_(min=0.0)
    minimax_model.r.data[overflow_idx] = r_max[overflow_idx]

    # dual update
    dual_loss = -(minimax_model.yloss() + minimax_model.ploss() + minimax_model.zloss(budget=args.budget))
    dual_optimizer.zero_grad()
    dual_loss.backward()


    a = 1
    dual_optimizer.step()

    proj_dual(minimax_model)



    if minimax_model.block_skip_gating is not None:
        out_gating = minimax_model.block_skip_gating.cpu().data.numpy()
    else:
        out_gating = None


    return cur_resource, minimax_model.s.cpu().data.numpy(), minimax_model.r.cpu().data.numpy(), out_gating, gating_grad_list



def uvc_optimizer_gating(optimizer, minimax_model, s_optimizer, r_optimizer, gating_optimizer, dual_optimizer, args, infos, save_budgets, flops_list):

    sr_loss2 = minimax_model.srloss2(budget = args.budget)
    cur_resource = sr_loss2.item() + args.budget

    dual_loss = -(minimax_model.zloss(budget=args.budget))
    dual_optimizer.zero_grad()
    dual_loss.backward()

    dual_optimizer.step()
    proj_dual(minimax_model)


    return cur_resource, minimax_model.s.cpu().data.numpy(), minimax_model.r.cpu().data.numpy(), minimax_model.block_skip_gating.cpu().data.numpy()


def build_minimax_model(model, layer_names, uvc_layers, uvc_layers_dict, args, flops_list, vanilla=False):
    minimax_model = UVC_CP_MiniMax(model,
                                  resource_fn=None,
                                  uvc_layers=uvc_layers,
                                  uvc_layers_dict=uvc_layers_dict,
                                  head_size=args.head_size,
                                  num_heads=args.num_heads,
                                  flops_list=flops_list,
                                  args=args)


    if args.flops_with_mhsa:


        cost_func = lambda s_,r_,ub_,gating,eps,gumbel_hard : calc_flops(s_, r_, uvc_layers_dict, uvc_layers, args.head_size, s_ub=minimax_model.s_ub, r_ub=minimax_model.r_ub, flops_list=flops_list, gating=gating, full_model_flops=ub_, eps=eps, use_gumbel=args.use_gumbel, gumbel_hard=gumbel_hard, args=args)
        resource_ub = float(
            cost_func(
                torch.zeros_like(minimax_model.s.data),
                torch.zeros_like(minimax_model.r.data),
                None,
                (None, None, None),
                0,
                True
            )
        )
        resource_fn = lambda s_,r_,gating,eps,gumbel_hard=False: cost_func(s_, r_, resource_ub, gating, eps,gumbel_hard)
        minimax_model.resource_fn = resource_fn


    else:
        cost_func = lambda s_,r_,ub_,flops_list : flops2(s_, r_, uvc_layers_dict, uvc_layers, args.head_size, ub=ub_, s_ub=minimax_model.s_ub, r_ub=minimax_model.r_ub, layer_names=layer_names, flops_list=flops_list)
        # resource rough overview
        resource_ub = float(cost_func(torch.zeros_like(minimax_model.s.data), torch.zeros_like(minimax_model.r.data),None,flops_list))
        # print()
        width_mult = [1, 0.75, 0.5, 0.25,0]
        for wm in width_mult:
            r_cost = float(cost_func(torch.round((1 - wm) * minimax_model.s_ub),torch.round((1 - wm) * minimax_model.r_ub),None, flops_list))
            # print('resource cost for {} model={:.8e}'.format(wm, r_cost))
        resource_fn = lambda s_,r_,flops_list: cost_func(s_, r_, resource_ub, flops_list)
        minimax_model.resource_fn = resource_fn
        # print(resource_fn(minimax_model.s_ub,minimax_model.r_ub))  # max remaining ratio after pruning

    minimax_model.model.enable_block_gating = args.enable_block_gating
    minimax_model.model.enable_part_gating  = args.enable_part_gating
    minimax_model.model.enable_patch_gating = args.enable_patch_gating
    minimax_model.model.enable_jumping      = args.enable_jumping
    minimax_model.model.use_gumbel          = args.use_gumbel
    minimax_model.model.eps                 = args.eps
    minimax_model.model.enable_warmpup      = args.enable_warmup


    print(f"** Initial FLOP size: {resource_ub/1e6:.2f}M")

    if vanilla:
        return minimax_model

    else:
        if args.soptim == 'adam':
            s_optimizer = torch.optim.Adam([minimax_model.s],
                                            args.slr,
                                            betas = (0.0, 0.999),
                                            weight_decay = 0.0)
        elif args.soptim == 'sgd':
            s_optimizer = torch.optim.SGD([minimax_model.s],
                                           args.slr,
                                           momentum = 0.0,
                                           weight_decay = 0.0)
        elif args.soptim == 'rmsprop':
            s_optimizer = torch.optim.RMSprop([minimax_model.s],
                                              lr = args.slr)
        else:
            raise NotImplementedError

        if args.roptim == 'adam':
            r_optimizer = torch.optim.Adam([minimax_model.r],
                                            args.rlr,
                                            betas = (0.0, 0.999),
                                            weight_decay = 0.0)
        elif args.roptim == 'sgd':
            r_optimizer = torch.optim.SGD([minimax_model.r],
                                           args.rlr,
                                           momentum = 0.0,
                                           weight_decay = 0.0)
        elif args.roptim == 'rmsprop':
            r_optimizer = torch.optim.RMSprop([minimax_model.r],
                                              lr = args.rlr)
        else:
            raise NotImplementedError

        if args.enable_block_gating:
            gating_optimizer = torch.optim.SGD([minimax_model.block_skip_gating],
                                           args.glr,
                                           momentum = 0.9,
                                           weight_decay = 1e-4)
        else:
            gating_optimizer = None



        dual_optimizer = torch.optim.SGD([{'params': minimax_model.z, 'lr': args.zlr_schedule_list[0]},
                                           {'params': minimax_model.y, 'lr': args.ylr},
                                          {'params': minimax_model.p, 'lr': args.plr}],
                                            1.0,
                                            momentum = 0.0,
                                            weight_decay = 0.0)

        return minimax_model, dual_optimizer, s_optimizer, r_optimizer, gating_optimizer

