# data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd_data_5h_clean_2.5h_noisy
data_cache=/home/stud_vantuan/share_with_150/cache/cd93_and_92_noisy_only_larger
checkpoints_dir=checkpoints/
gpu_ids=0
config=ngf64_unet128_cd93_to_cd92_noise_only_larger_200epochs

# unet_128_mask, unet_256_mask

python train.py \
    --dataroot $data_cache \
    --name unet_gan_${config} \
    --model unet_gan \
    --model_name unet_128_mask \
    --dataset_mode audio \
    --pool_size 50 \
    --no_dropout \
    --norm instance \
    --lambda_A 10 \
    --lambda_B 10 \
    --lambda_identity 0.5 \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --preprocess none \
    --batch_size 1 \
    --lr_G 0.0001 \
    --lr_D 0.0001 \
    --G-update-frequency 1 \
    --D-update-frequency 1 \
    --constant-gp 1 \
    --lambda-gp 0.05 \
    --ngf 64 \
    --niter 100 \
    --niter_decay 100 \
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

python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/cd93_100k_line \
    --name unet_gan_ngf64_unet128_cd93_to_cd92_noise_only_larger_200epochs \
    --model unet_gan \
    --dataset_mode audio \
    --ngf 64 \
    --norm instance \
    --phase test \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --no_dropout \
    --batch_size 1 \
    --gpu_ids $gpu_ids \
    --input_nc 1 \
    --output_nc 1 \
    --raw-feat \
    --use_mask