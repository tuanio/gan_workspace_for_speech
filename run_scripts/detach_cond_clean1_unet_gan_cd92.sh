# data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd_data_5h_clean_2.5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd93_1h_subset_cd92_cluster46_over100
data_cache=/home/stud_vantuan/share_with_150/cache/cd92_10h_9h_cond
checkpoints_dir=checkpoints/
gpu_ids=2
config=cond_detach_dim32_unet128_label.clean1_noisy100_10h_9h_100epochs

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
    --batch_size 1 \
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
    --label_A_path /home/stud_vantuan/share_with_150/cache/cd92_10h_9h_cond/label/KMeans_train_clean_cd93_1.clusters \
    --label_B_path /home/stud_vantuan/share_with_150/cache/cd92_10h_9h_cond/label/KMeans_train_noisy_cd92_100.clusters \
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

# ---
python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/cd92_10h_9h_cond \
    --name cond_unet_gan_cond_dim32_unet128_label.clean1_noisy100_10h_9h_100epochs \
    --model conditional_gan \
    --model_name unet_128_mask \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --no_dropout \
    --batch_size 1 \
    --gpu_ids 2 \
    --input_nc 3 \
    --output_nc 1 \
    --raw-feat \
    --use_mask \
    --apply_spectral_norm \
    --conditional \
    --label_embed_dim 32 \
    --num_labels_A 1 \
    --num_labels_B 100 \
    --label_B_path /home/stud_vantuan/share_with_150/cache/cd92_10h_9h_cond/label/generated_test_noisy_cd93_based_on_cd92_100.clusters

# 
# python test.py \
#     --dataroot /home/stud_vantuan/share_with_150/cache/cd92_10h_9h_cond \
#     --name cond_unet_gan_cond_dim32_unet128_label.clean100_noisy100_10h_9h_200epochs \
#     --model conditional_gan \
#     --model_name unet_128_mask \
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
#     --raw-feat \
#     --use_mask \
#     --apply_spectral_norm \
#     --conditional \
#     --label_embed_dim 32 \
#     --num_labels_A 100 \
#     --num_labels_B 100 \
#     --label_B_path /home/stud_vantuan/share_with_150/cache/cd92_10h_9h_cond/label/generated_test_noisy_cd93_based_on_cd92_100.clusters