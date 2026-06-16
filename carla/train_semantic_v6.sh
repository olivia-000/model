#!/bin/bash
# Train v6: Pure NuRec 30,775 + Mask2Former labels, from scratch
# Run from: /home/itriu100/carla/

cd pix2pixHD

python train.py \
  --name carla2real_semantic_v6 \
  --dataroot /home/itriu100/carla/datasets/training_semantic_v6 \
  --label_nc 19 \
  --no_instance \
  --loadSize 1024 \
  --fineSize 512 \
  --batchSize 4 \
  --niter 100 \
  --niter_decay 100 \
  --save_epoch_freq 10 \
  --lr 0.0002 \
  --resize_or_crop scale_width_and_crop \
  --gpu_ids 0
