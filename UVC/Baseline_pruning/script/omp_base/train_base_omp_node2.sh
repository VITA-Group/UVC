export MASTER_ADDR=10.124.136.83
export MASTER_PORT=20335
export NCCL_SOCKET_IFNAME='enp179s0f0'
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=3 \
    --node_rank=2 \
    --master_addr='10.124.136.83' \
    --master_port=20335 \
    --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 256 \
    --data-path /datadrive_d/TLC/imagenet \
    --init_mask deit_mask/deit_base_patch16_224_omp50_mag.pt \
    --output_dir experiment/deit_base_omp_50



