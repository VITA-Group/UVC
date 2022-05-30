python -u generating_mask.py \
    --sparsity 0.4 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
    --model deit_small_patch16_224 \
    --save_file deit_small_patch16_224_omp40_mag.pt \
    --type mag 
