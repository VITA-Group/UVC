python -u generating_mask.py \
    --sparsity 0.4 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth \
    --model deit_base_patch16_224 \
    --save_file deit_base_patch16_224_taylor1score40_mag.pt \
    --data $1 \
    --type taylor