data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd_data_5h_clean_2.5h_noisy
checkpoints_dir=checkpoints/
gpu_ids=0
config=cycle50_idt5_sn_gp2_lr0.0002_512fft_5h_5h_200epochs

python train.py \
    --dataroot $data_cache \
    --name unet_gan_${config} \
    --model unet_gan \
    --dataset_mode audio \
    --pool_size 100 \
    --no_dropout \
    --norm instance \
    --lambda_A 50 \
    --lambda_B 50 \
    --lambda_identity 5 \
    --n_fft 512 \
    --hop_length 64 \
    --fix_w 257 \
    --load_size_h 257 \
    --load_size_w 257 \
    --crop_size 257 \
    --preprocess none \
    --batch_size 4 \
    --lr_G 0.0002 \
    --lr_D 0.0002 \
    --D-update-frequency 1 \
    --D-update-frequency 1 \
    --constant-gp 1 \
    --apply_spectral_norm \
    --lambda-gp 2 \
    --niter 200 \
    --niter_decay 200 \
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
    --num_threads 2 \
    --use-wandb \
    --wandb-project GAN_for_CD92 \
    --wandb-run-name unet_gan_${config}

#
python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/cd92.93_95_old_with_1h_clean_and_30m_noisy \
    --name unet_gan_cycle50_idt5_sn_gp2_lr0.0002_512fft_5h_5h_200epochs \
    --model unet_gan \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --n_fft 512 \
    --hop_length 64 \
    --fix_w 257 \
    --load_size_h 257 \
    --load_size_w 257 \
    --crop_size 257 \
    --batch_size 1 \
    --gpu_ids 0 \
    --input_nc 1 \
    --output_nc 1 \
    --raw-feat \
    --apply_spectral_norm \
    --use_mask

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