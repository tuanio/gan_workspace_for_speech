# this one train viunet with raw feat from 5h
data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_with_5h_clean_and_5h_noisy
# data_cache=/home/stud_vantuan/share_with_150/cache/cd92.93_95_old_with_1h_clean_and_30m_noisy
checkpoints_dir=checkpoints/
gen_pretrained_path=/home/stud_vantuan/projects/aug_asr/gan_workspace_for_speech/checkpoints/vit_unet_pretrain/latest_net_G.pth
gpu_ids=0
config=pretrain_raw_feat

python train.py \
    --dataroot $data_cache \
    --name vit_unet_${config} \
    --model vit_unet \
    --dataset_mode audio \
    --pool_size 50 \
    --no_dropout \
    --norm instance \
    --lambda_A 10 \
    --lambda_B 10 \
    --lambda_identity 0.5 \
    --constant-gp 100 \
    --lambda-gp 0.1 \
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
    --gen-pretrained-path $gen_pretrained_path \
    --wandb-project GAN_for_CD92 \
    --wandb-run-name vit_unet_${config}
