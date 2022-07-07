
python -u generating_mask.py \
    --type sp \
    --atten_density 0.7 \
    --mlp_density 0.6 \
    --heads 6 \
    --batch_size 256 \
    --pretrained https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
    --model deit_small_patch16_224_sp \
    --save_file deit_small_patch16_224_sp60.pt \
    --data $1
