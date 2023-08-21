set -ex
python test.py \
    --dataroot /data/tuanio/data/cache/SpeechAttentionGAN-VC/data_cache \
    --name speech_attention_gan_$1 \
    --model attention_gan \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 128 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids 2 \
    --input_nc 1 \
    --output_nc 1 \
    --use_mask