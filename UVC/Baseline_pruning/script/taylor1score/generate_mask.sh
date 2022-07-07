python -u generating_mask.py \
    --sparsity 0.5 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth \
    --model deit_base_patch16_224 \
    --save_file deit_base_patch16_224_taylor1score50_mag.pt \
    --data $1

python -u generating_mask.py \
    --sparsity 0.5 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
    --model deit_small_patch16_224 \
    --save_file deit_small_patch16_224_taylor1score50_mag.pt \
    --data $1

python -u generating_mask.py \
    --sparsity 0.6 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
    --model deit_tiny_patch16_224 \
    --save_file deit_tiny_patch16_224_taylor1score60_mag.pt \
    --data $1
