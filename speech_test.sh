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
---

python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/cd92.93_95_old_with_1h_clean_and_30m_noisy \
    --name unet_gan_raw_feat_5h_5h_150epochs \
    --model unet_gan \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids 2 \
    --input_nc 1 \
    --output_nc 1 \
    --raw-feat \
    --use_mask

python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/add_10h_cd93.95 \
    --name unet_gan_raw_feat_5h_5h_150epochs \
    --model unet_gan \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids 3 \
    --input_nc 1 \
    --output_nc 1 \
    --raw-feat \
    --use_mask


---

python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/cd92.93_95_old_with_1h_clean_and_30m_noisy \
    --name vit_unet_pretrain_raw_feat_5h_5h_150epochs \
    --model vit_unet \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids 3 \
    --input_nc 1 \
    --output_nc 1 \
    --raw-feat \
    --use_mask

--- 

python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/cd92.93_95_old_with_1h_clean_and_30m_noisy \
    --name conformer_unet_with_pretrain_raw_feat_5h_5h_150epochs \
    --model conformer_unet \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 129 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids 3 \
    --input_nc 1 \
    --output_nc 1 \
    --raw-feat \
    --use_mask

---

python test.py \
    --dataroot /home/stud_vantuan/share_with_150/cache/cd92.93_95_old_with_1h_clean_and_30m_noisy \
    --name attn_gan_old_pipe_5h_5h_150epochs \
    --model attention_gan \
    --dataset_mode audio \
    --norm instance \
    --phase test \
    --no_dropout \
    --load_size_h 128 \
    --load_size_w 128 \
    --crop_size 128 \
    --batch_size 1 \
    --gpu_ids 3 \
    --input_nc 1 \
    --output_nc 1 \
    --use_mask