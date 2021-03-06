python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port 6382 post_train.py \
    --pretrained 0 \
    --model_type "deit_tiny_patch16_224" \
    --model_path https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth \
    --checkpoint_dir /home/sy22725/UVC_timm/UVC/mc_deit_tiny_224/debug/deit_tiny_patch16_224_9.pth.tar \
    --data_dir "/ssd1/shixing/imagenet2012" \
    --distillation-type soft \
    --distillation-alpha 0.1 \
    --train_batch_size 128 \
    --gpu_num '1,2' \
    --epochs 120 \
    --eval_every 1000 \
    --output_dir finetuning/ft_deit_tiny_patch16_224_0.48 \
    --num_workers 64 \
    --learning_rate 1e-4 |  tee ft_deit_tiny_0.5.log
