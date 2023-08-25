# data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd_data_5h_clean_2.5h_noisy
data_cache=/home/stud_vantuan/share_with_150/cache/cd93_1h_subset_cd92_cluster46_over100
checkpoints_dir=checkpoints/
gpu_ids=3
config=cluster46_1h.clean_13m.noisy_200epochs

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
    --lambda_identity 1 \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --preprocess none \
    --batch_size 8 \
    --lr_G 0.0002 \
    --lr_D 0.0002 \
    --G-update-frequency 1 \
    --D-update-frequency 1 \
    --constant-gp 1 \
    --lambda-gp 0.1 \
    --apply_spectral_norm \
    --niter 60 \
    --niter_decay 60 \
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
#     --lambda_A 5 \
#     --lambda_B 5 \
#     --lambda_identity 1 \
#     --load_size_h 129 \
#     --load_size_w 128 \
#     --crop_size 128 \
#     --preprocess none \
#     --batch_size 4 \
#     --threshold-to-cut 30 \
#     --minimum-start-end 100 \
#     --cut-noisy \
#     --lr_G 0.0002 \
#     --lr_D 0.0002 \
#     --G-update-frequency 1 \
#     --D-update-frequency 1 \
#     --constant-gp 1 \
#     --lambda-gp 1.0 \
#     --niter 100 \
#     --niter_decay 100 \
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
#     --num_threads 1 \
#     --use-wandb \
#     --wandb-project GAN_for_CD92 \
#     --wandb-run-name unet_gan_${config}

# # 
# python test.py \
#     --dataroot /home/stud_vantuan/share_with_150/cache/cd92.93_95_old_with_1h_clean_and_30m_noisy \
#     --name unet_gan_cut30_thres100_cut_noisy_feat_lrG0.0002_lr_D0.0002_updateD_faster_2times_GP_5h_2.5h_100epochs \
#     --model unet_gan \
#     --dataset_mode audio \
#     --norm instance \
#     --phase test \
#     --load_size_h 129 \
#     --load_size_w 128 \
#     --crop_size 128 \
#     --no_dropout \
#     --batch_size 1 \
#     --gpu_ids 3 \
#     --input_nc 1 \
#     --output_nc 1 \
#     --use_mask

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