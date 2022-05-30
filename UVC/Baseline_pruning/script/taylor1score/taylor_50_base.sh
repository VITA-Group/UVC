python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 256 \
    --data-path $1 \
    --init_mask deit_mask/deit_base_patch16_224_taylor1score50_mag.pt \
    --output_dir experiment/taylor1score_4gpus_deit_base_sparsity50 \
    --dist_url 'tcp://127.0.0.1:33251'