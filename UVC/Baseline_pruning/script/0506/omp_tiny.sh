
# python -u generating_mask.py \
#     --sparsity 0.7 \
#     --pretrained https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
#     --model deit_tiny_patch16_224 \
#     --save_file deit_mask/deit_tiny_patch16_224_omp70_mag.pt


python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_tiny_patch16_224 \
    --batch-size 256 \
    --data-path $1 \
    --init_mask deit_mask/deit_tiny_patch16_224_omp70_mag.pt \
    --output_dir experiment/omp_4gpus_deit_tiny_sparsity70 \
    --dist_url 'tcp://127.0.0.1:33251' \
    --num_workers $2