python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 128 \
    --data-path $1 \
    --init_mask deit_base_patch16_224_omp40_mag.pt \
    --output_dir experiment/omp_8gpus_deit_base40 \
    --dist_url 'tcp://127.0.0.1:24568' \
    --num_workers $2