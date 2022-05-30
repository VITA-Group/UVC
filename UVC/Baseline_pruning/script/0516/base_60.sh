python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 128 \
    --data-path $1 \
    --init_mask deit_base_patch16_224_sp60.pt \
    --output_dir experiment/sp_base_60 \
    --dist_url 'tcp://127.0.0.1:33251' \
    --num_workers $2