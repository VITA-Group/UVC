## UVC

**[Undate from Tianlong]**

Codes for baseline pruning methods are in the "Baseline_pruning" folder

## Set Env

The same as the DeiT, refer to [README.md](./Baseline_pruning/README_DeIT.md)

## Commands

1. One-shot Magnitude Pruning (OMP)

Refer to scripts in folder ```./Baseline_pruning/script/omp_base```

2. Talyer Pruning (TP)

Refer to scripts in folder ```./Baseline_pruning/script/taylor1score```

3. Gradually Magnitude Pruning (GMP)

Refer to scripts in folder ```./Baseline_pruning/script/gmp```

4. Structure Pruning (SP)

Coming soon!

## Pre-trained Models

1. GMP on DeiT-Base

https://www.dropbox.com/s/4j0fkkazcrw14ry/deit_base_GMP_50_80.79.pth?dl=0

2. TP on DeiT-Base

https://www.dropbox.com/s/itio3qrd1qtuaoy/deit_base_TP_50_80.55.pth?dl=0

3. SP on DeiT-Base

https://www.dropbox.com/s/d72aoeq18ihvdlo/deit_base_SP_40_80.08.pth?dl=0









## Update
[From Shixing]

New function:

#### uvc_utils.py

* Newly added function: calc_flops
 * MHSA flops are added.
 * Remember to make it a function so the inplace tensor won’t disappear after two backwards
 * Remember to let every part of the torch.grad to participate in calculation



* Newly added function: PresetLRScheduler:
 * To control zlr in a group of parameters





#### uvc_optimizer.py:

* build_minimax_model
 * Newly added resource_fn2, which is the function in uvc_utils.py: calc_flops





#### models/model_distilled.py

The whole structure of DeiT and ViT, convenient to calculate the FLOPs of the full model.

The forward code of block is very important and easy to make mistake when calculate flops:

Wrong way:

```python
def forward(self, x):

   attention, macs1 = self.attn(self.norm1(x))
   x = x + self.drop_path(attention)

   x, macs2 = self.mlp(self.norm2(x))
   x = x + self.drop_path(x)
   return x, macs1 + macs2
```



Right way:

```python
def forward(self, x):
   shortcut = x
   x, macs_attn = self.attn(self.norm1(x))
   x = shortcut + self.drop_path(x)

   shortcut = x
   x, macs_mlp = self.mlp(self.norm2(x))
   x = shortcut + self.drop_path(x)
   return x, macs_attn + macs_mlp
```



#### models/model_edit.py

Tryouts of different ways to calculate flops of DeiT





#### joint_train.py

* Added zlr_schedule
* Code for post training
* Code for DeiT and the whole set of fitted training setting
* Add binary mask to the original model


#### post_train.py

* Stage 2 training to improve accuracy.



#### Run Stage1 UVC Training
```bash
python -m torch.distributed.launch --nproc_per_node=2 joint_train.py --uvc_train --model_type "deit_tiny_distilled_patch16_224"  --train_batch_size 512 --num_epochs 20 --eval_every 1000 --flops_with_mhsa 1 --zlr_schedule_list "10,20,30,40,50" --learning_rate 1e-4 --budget 0.5
```

#### Run Stage2 Post Training
```bash
python -m torch.distributed.launch --nproc_per_node=2 post_train.py --checkpoint_dir "/home/shixing/UVC_vita4/output/debug/checkpoint_resource_0.5.bin" --model_type "deit_tiny_distilled_patch16_224"  --train_batch_size 512 --num_epochs 100 --eval_every 1000 --post_learning_rate 1e-4
```
