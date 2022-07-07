python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --model deit_small_patch16_224_data \
    --batch-size 128 \
    --data-path $1 \
    --output_dir experiment/deit_small_8gpus_bs128x8_layer_wise_token177 \
    --dist_url 'tcp://127.0.0.1:33251' \
    --token_selection \
    --token_number 177 \
    --num_workers $2