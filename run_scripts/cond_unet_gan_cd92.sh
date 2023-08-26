# data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd_data_5h_clean_2.5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd93_1h_subset_cd92_cluster46_over100
data_cache=/home/stud_vantuan/share_with_150/cache/cd93_10h_9h
checkpoints_dir=checkpoints/
gpu_ids=2
config=cond_dim32_unet128_label.clean1_noisy100_10h_9h_400epochs

# unet_128_mask, unet_256_mask

python train.py \
    --dataroot $data_cache \
    --name cond_unet_gan_${config} \
    --model conditional_gan \
    --model_name unet_128_mask \
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
    --conditional \
    --label_embed_dim 32 \
    --num_labels_A 1 \
    --num_labels_B 100 \
    --label_A_path  \
    --label_B_path  \
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
    --num_threads 4 \
    --use-wandb \
    --wandb-project GAN_for_CD92 \
    --wandb-run-name unet_gan_${config}