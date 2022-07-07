
##
from __future__ import absolute_import, division, print_function

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
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import *
from models.modeling import VisionTransformer, CONFIGS
from models.model_distilled import DistilledVisionTransformer

from functools import partial
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, PresetLRScheduler
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.losses import DistillationLoss
from uvc_optimizer import uvc_optimizer, build_minimax_model
import json
import copy
from uvc_utils import prune_w

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from T2TViT.models.t2t_vit import *
from T2TViT.utils import load_for_transfer_learning



import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


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
    print("Saved model checkpoint to [DIR: %s]", args.output_dir)


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
        # model = DistilledVisionTransformer(config.embed_dim, config, args.img_size, zero_head=True, num_classes=args.num_classes)
        # model.default_cfg = _cfg()

        model = DistilledVisionTransformer(
            enable_dist=args.enable_deit,
            patch_size=config.patch_size, embed_dim=config.embed_dim, depth=config.depth, num_heads=config.num_heads, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0,
            gumbel_hard=True
        )
        for name, p in model.named_modules():
            if hasattr(p, "weight"):
                p.register_buffer("mask", torch.ones_like(p.weight).cuda())
        if args.pretrained == True:
            checkpoint = torch.load(f"../pretrained_models/deit/{args.model_type}.pth",
                map_location="cpu"
            )
            model.load_state_dict(checkpoint["model"], strict=False)

    elif "t2t" in args.model_type:
        # create model
        model = t2t_vit_14()
        for name, p in model.named_modules():
            if hasattr(p, "weight"):
                p.register_buffer("mask", torch.ones_like(p.weight))

        # load the pretrained weights
        load_for_transfer_learning(model, args.model_path, use_ema=True, strict=False, num_classes=1000)
    else:
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes)
    # model.load_from(np.load(args.pretrained_dir))


    model.to(args.device)
    num_params = count_mask(model)

    # print("{}".format(config))
    print("Training parameters %s", args)
    print("Total Parameter: \t%2.1fM" % num_params)
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

    print("***** Running Validation *****")
    print("  Num steps = %d", len(test_loader))
    print("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        for name, m in model.named_modules():
            if hasattr(m, "mask"):
                # print(name, m)
                m.weight.data *= m.mask
        with torch.no_grad():
            if args.distillation_type == None:
                logits = model(x)[0]
            else:
                logits, flops_list = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    print("\n")
    print("Validation Results")
    print("Global Steps: %d" % global_step)
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % accuracy)
    if args.enable_writer:
        writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy




def post_training(args, model, mixup_fn=None, criterion=None):
    writer=None
    if args.local_rank in [-1, 0, 1]:
        os.makedirs(os.path.join(args.output_dir,args.name), exist_ok=True)
        if args.enable_writer:
            writer = SummaryWriter(log_dir=os.path.join("post_train", args.name))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    print("Starting post training")
    train_loader, test_loader = get_loader(args)

    t_total = len(train_loader) * args.epochs # modify
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    print("[before DDP]")
    model_without_ddp = model
    if args.local_rank!=-1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size(), delay_allreduce=True)
        model_without_ddp = model.module
    else:
        model = nn.DataParallel(model)


    linear_scaled_lr = args.learning_rate * args.train_batch_size * get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    scheduler, _ = create_scheduler(args, optimizer)

    # Train!
    print("***** [Stage 2] Post Training *****")
    print("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.module.block_skip_gating.requires_grad = False
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    # global_step, best_acc = 0, 0
    global_step = 0
    best_acc = 0

    epoch_iterator = tqdm(train_loader,
                  desc="Training [X / X Steps] [LR: X | Loss: X.XXX]",
                  # desc="Training [X / X Steps] [loss = X.X]",
                  bar_format="{l_bar}{r_bar}",
                  dynamic_ncols=True)


    delta1_log = []
    delta2_log = []








    for epoch in range(args.epochs):
        model.train()
        print("="*60)
        print(f"Start training [Epoch {epoch}]")

        model.module.block_skip_gating.requires_grad = False

        epoch_iterator = tqdm(train_loader,
                      desc="Training [X / X Steps] [LR: X | Loss: X.XXX]",
                      # desc="Training [X / X Steps] [loss = X.X]",
                      bar_format="{l_bar}{r_bar}",
                      dynamic_ncols=True)
        start_time = time.time()
        scheduler.step(epoch)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.cuda() for t in batch)
            x, y = batch
            if len(x) % 2 != 0:
                x = x[:-1]
                y = y[:-1]
            for name, m in model.named_modules():
                if hasattr(m, "mask"):
                    # print(name, m)
                    m.weight.data *= m.mask

            x, y = mixup_fn(x, y)
            outputs, flops_list = model(x)
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


                global_step += 1
                optimizer.zero_grad()


                epoch_iterator.set_description(
                    "Training [%d / %d Steps] [LR: %.6f | Loss: %.3f]" % (global_step, t_total, scheduler.get_epoch_values(epoch)[0], losses.val)
                )

                if args.local_rank in [-1, 0] and args.enable_writer:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_epoch_values(epoch)[0], global_step=global_step)


        if args.local_rank in [-1, 0]:
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
    parser.add_argument("--data_dir", default="/ssd1/xinyu/dataset/imagenet2012",
                        help="directory of data")

    parser.add_argument("--num_workers", default=8, type = int)
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "t2t_vit_14"] + deit_family,
                        default="deit_tiny_distilled_patch16_224",
                        help="Which variant to use.")
    parser.add_argument("--model_path", default="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                        help="State dict path")


    parser.add_argument("--pretrained", type=int, default=0,
                        help="Whether we use a pretrained ViT models.")
    parser.add_argument("--output_dir", default="output_post", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")


    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for SGD.")

    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")


    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')


    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')



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
    parser.add_argument('--teacher-model', default='', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    parser.add_argument("--use_distribute", default=0, type=int,
                        help="Whether launch distributed learning")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/shixing/UVC_all/UVC_vita4/output/uvc_train/debug/deit_tiny_distilled_patch16_224_19.pth.tar",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--gpu_num", type=str, default="0,1",
                        help="Used gpu number")
    parser.add_argument("--enable_writer", default=0, type=int,
                        help="Using or not using tensorboard writer")
    parser.add_argument("--enable_jumping", type=int, default=0,
                        help="Whether jump the past connection to the last layer")
    parser.add_argument("--enable_deit", type=int, default=0,
                        help="whether use deit structure")

    args = parser.parse_args()

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

    args, model = setup(args)
    teacher_model = None

    # Set up mix-up function

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
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
        if args.teacher_model == "":
            args.teacher_model = args.model_type
        if args.teacher_path == "":
            args.teacher_path  = args.model_path
        print(f"** Creating teacher model: [{args.teacher_model}] loading from ==> [{args.teacher_path}]")
        if 'deit' in args.teacher_model:
            teacher_model = DistilledVisionTransformer(
                enable_dist=args.enable_deit,
                patch_size=config.patch_size, embed_dim=config.embed_dim, depth=config.depth, num_heads=config.num_heads, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0,
            )


            if args.teacher_path is not None: # 'need to specify teacher-path when using distillation'
                if args.teacher_path.startswith('https'):
                    checkpoint = torch.hub.load_state_dict_from_url(
                        args.teacher_path, map_location='cpu', check_hash=True)
                else:
                    checkpoint = torch.load(args.teacher_path, map_location='cpu')
                teacher_model.load_state_dict(checkpoint['model'], strict=False)


        elif "t2t" in args.teacher_model:
            teacher_model = t2t_vit_14(pretrained=True)
            if args.local_rank in [-1, 0]:
                print(f"-----Has teacher_model [{args.teacher_model}]-----")
            load_for_transfer_learning(model, args.model_path, use_ema=True, strict=False, num_classes=1000)


        teacher_model.to(device)
        teacher_model.eval()
    # Construct distillation loss
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    if args.distillation_type == 'none':
        args.distillation_type = None


    print('==> Loading checkpoint from %s.' % args.checkpoint_dir)
    checkpoint = torch.load(args.checkpoint_dir, map_location='cpu')
    if hasattr(checkpoint, 'args'):
        _, state_dict, s, r = checkpoint["args"], checkpoint["model"], checkpoint["s"], checkpoint["r"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)



    post_training(args, model, mixup_fn, criterion)





if __name__ == "__main__":
    main()