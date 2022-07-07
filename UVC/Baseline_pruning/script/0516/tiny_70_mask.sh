python -u generating_mask.py \
    --type sp \
    --atten_density 0.7 \
    --mlp_density 0.7 \
    --heads 3 \
    --batch_size 256 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
    --model deit_tiny_patch16_224_sp \
    --save_file deit_tiny_patch16_224_sp70.pt \
    --data $1
