# $1 can be helicopter, cabin, codec2

noise_type=helicopter
src_type=librispeech # timit or librispeech
gpu_ids=0,1,2,3
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
    --name speech_attention_gan_${src_type}_${noise_type} \
    --model attention_gan \
    --dataset_mode audio \
    --pool_size 50 \
    --no_dropout \
    --norm instance \
    --lambda_A 10 \
    --lambda_B 10 \
    --lambda_identity 0.5 \
    --load_size_h 128  \
    --load_size_w 128  \
    --crop_size 128 \
    --preprocess none \
    --batch_size 16 \
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
    --grayscale \
    --max_mask_len 50 \
    --checkpoints_dir $checkpoints_dir \
    --no_html \
    --num_threads 8
fi

if [ $stage -le 2 ]; then
python test.py \
    --dataroot $data_cache \
    --name speech_attention_gan_${src_type}_${noise_type} \
    --model attention_gan \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 128 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids $gpu_ids \
    --input_nc 1 \
    --output_nc 1 \
    --use_mask
fi


# python cal_metrics.py \
#    --data-cache $data_cache \
#    --results-dir results/ \
#    --name speech_attention_gan_${src_type}_$noise_type
