python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --model deit_tiny_patch16_224_8layer \
    --batch-size 128 \
    --data-path $1 \
    --output_dir experiment/8layers_8gpus_deit_tiny \
    --dist_url 'tcp://127.0.0.1:34873' \
    --num_workers $2