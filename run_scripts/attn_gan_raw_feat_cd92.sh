data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
checkpoints_dir=checkpoints/
gpu_ids=0,1
config=raw_feat_5h_5h_150epochs

python train.py \
    --dataroot $data_cache \
    --name attn_gan_${config} \
    --model attention_gan \
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
    --batch_size 16 \
    --niter 100 \
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
    --num_threads 8 \
    --use-wandb \
    --wandb-project GAN_for_CD92 \
    --wandb-run-name attn_gan_${config}
