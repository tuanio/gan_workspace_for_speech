data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd_data_5h_clean_2.5h_noisy
checkpoints_dir=checkpoints/
gpu_ids=1
config=raw_feat_512fft_lrG0.0002_lr_D0.0001_updateD_slower_5times_GP_5h_5h_100epochs

python train.py \
    --dataroot $data_cache \
    --name unet_gan_${config} \
    --model unet_gan \
    --dataset_mode audio \
    --pool_size 50 \
    --no_dropout \
    --norm instance \
    --lambda_A 10 \
    --lambda_B 10 \
    --lambda_identity 0.5 \
    --n_fft 512 \
    --hop_length 32 \
    --fix_w 257 \
    --load_size_h 257 \
    --load_size_w 257 \
    --crop_size 257 \
    --preprocess none \
    --batch_size 8 \
    --lr_G 0.0002 \
    --lr_D 0.0001 \
    --D-update-frequency 5 \
    --constant-gp 100 \
    --lambda-gp 0.1 \
    --niter 50 \
    --niter_decay 50 \
    --gpu_ids $gpu_ids \
    --display_id 0 \
    --display_freq 200 \
    --print_freq 200 \
    --input_nc 1 \
    --output_nc 1 \
    --use_cycled_discriminators \
    --use_mask \
    --raw-feat \
    --max_mask_len 50 \
    --checkpoints_dir $checkpoints_dir \
    --no_html \
    --num_threads 4 \
    --use-wandb \
    --wandb-project GAN_for_CD92 \
    --wandb-run-name unet_gan_${config}

# python train.py \
#     --dataroot $data_cache \
#     --name unet_gan_${config} \
#     --model unet_gan \
#     --dataset_mode audio \
#     --pool_size 50 \
#     --no_dropout \
#     --norm instance \
#     --lambda_A 10 \
#     --lambda_B 10 \
#     --lambda_identity 0.5 \
#     --load_size_h 129 \
#     --load_size_w 128 \
#     --crop_size 128 \
#     --preprocess none \
#     --batch_size 32 \
#     --niter 100 \
#     --niter_decay 50 \
#     --gpu_ids $gpu_ids \
#     --display_id 0 \
#     --display_freq 200 \
#     --print_freq 200 \
#     --input_nc 1 \
#     --output_nc 1 \
#     --use_cycled_discriminators \
#     --use_mask \
#     --raw-feat \
#     --max_mask_len 50 \
#     --checkpoints_dir $checkpoints_dir \
#     --no_html \
#     --num_threads 8 \
#     --use-wandb \
#     --wandb-project GAN_for_CD92 \
#     --wandb-run-name unet_gan_${config}