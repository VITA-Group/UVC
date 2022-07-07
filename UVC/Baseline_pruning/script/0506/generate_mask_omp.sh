python -u generating_mask.py \
    --sparsity 0.7 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
    --model deit_tiny_patch16_224 \
    --save_file deit_mask/deit_tiny_patch16_224_omp70_mag.pt

# python -u generating_mask.py \
#     --sparsity 0.7 \
#     --pretrained https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
#     --model deit_tiny_patch16_224 \
#     --save_file deit_mask/deit_tiny_patch16_224_taylor1score70_mag.pt \
#     --data $1
