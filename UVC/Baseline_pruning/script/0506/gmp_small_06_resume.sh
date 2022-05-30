export NCCL_P2P_DISABLE=1
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_small_patch16_224 \
    --batch-size 256 \
    --data-path $1 \
    --output_dir experiment/gmp_4gpus_deit_small_density40_new \
    --dist_url 'tcp://127.0.0.1:23454' \
    --gmp \
    --sparsity 0.6 \
    --pruning_times 20 \
    --delta_t 6250 \
    --t_start 62500 \
    --num_workers $2 \
    --resume experiment/gmp_4gpus_deit_small_density40_new/checkpoint.pth 