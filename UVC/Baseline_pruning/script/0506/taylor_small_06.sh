python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_small_patch16_224 \
    --batch-size 256 \
    --data-path $1 \
    --init_mask deit_small_patch16_224_taylor1score40_mag.pt \
    --output_dir experiment/taylor1score_4gpus_deit_small_density40 \
    --dist_url 'tcp://127.0.0.1:33251' \
    --num_workers $2