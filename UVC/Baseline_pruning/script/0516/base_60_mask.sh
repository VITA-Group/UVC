python -u generating_mask.py \
    --type sp \
    --atten_density 0.7 \
    --mlp_density 0.6 \
    --heads 12 \
    --batch_size 128 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth \
    --model deit_base_patch16_224_sp \
    --save_file deit_base_patch16_224_sp60.pt \
    --data $1
