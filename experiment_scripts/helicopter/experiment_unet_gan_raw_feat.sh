# $1 can be helicopter, cabin, codec2

noise_type=helicopter
src_type=librispeech # timit or librispeech
gpu_ids=2,3
stage=1
data_cache=/home/stud_vantuan/share_with_150/cache_113/$src_type/$noise_type
checkpoints_dir=checkpoints/

# python -m datasets.fetchData \
#    --timit-dir /data/tuanio/data/noise/TIMIT_$noise_type \
#    --data_cache $data_cache \
#    --timit_noise_type $noise_type \
#    --transfer_mode timit

if [ $stage -le 1 ]; then
python train.py \
    --dataroot $data_cache \
    --name unet_gan_raw_feat_${src_type}_${noise_type} \
    --model unet_gan \
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
    --niter 150 \
    --niter_decay 150 \
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
    --wandb-project speech_attn_gan_helicopter \
    --wandb-run-name unet_gan_raw_feat
fi

if [ $stage -le 2 ]; then
python test.py \
    --dataroot $data_cache \
    --name unet_gan_raw_feat_${src_type}_${noise_type} \
    --model unet_gan \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids $gpu_ids \
    --input_nc 1 \
    --output_nc 1 \
    --raw-feat \
    --use_mask
fi


# python cal_metrics.py \
#    --data-cache $data_cache \
#    --results-dir results/ \
#    --name speech_attention_gan_${src_type}_$noise_type
