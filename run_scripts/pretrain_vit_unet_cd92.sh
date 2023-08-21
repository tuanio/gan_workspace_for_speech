data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
checkpoints_dir=checkpoints/
gpu_ids=1
config=pretrain

python pretrain.py \
    --dataroot $data_cache \
    --name vit_unet_${config} \
    --model pretrain_generator \
    --model-name vit_unet_mask \
    --dataset_mode audio \
    --pool_size 50 \
    --no_dropout \
    --norm instance \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --preprocess none \
    --batch_size 64 \
    --niter 200 \
    --niter_decay 200 \
    --gpu_ids $gpu_ids \
    --display_id 0 \
    --display_freq 200 \
    --print_freq 200 \
    --input_nc 1 \
    --output_nc 1 \
    --use_mask \
    --raw-feat \
    --max_mask_len 50 \
    --checkpoints_dir $checkpoints_dir \
    --no_html \
    --num_threads 8 \
    --is-pretrain \
    --time-width 25 \
    --time-masks 2 \
    --freq-width 25 \
    --freq-masks 2 \
    --beta1 0.9 \
    --lr 0.00062 \
    --use-wandb \
    --wandb-project GAN_for_CD92 \
    --wandb-run-name vit_unet_${config}