python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_small_patch16_224_half \
    --batch-size 256 \
    --data-path $1 \
    --output_dir experiment/6layers_4gpus_deit_small \
    --dist_url 'tcp://127.0.0.1:39277' \
    --num_workers $2