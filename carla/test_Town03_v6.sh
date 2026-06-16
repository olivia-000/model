#!/bin/bash
# Inference v6 model on Town03 (Mask2Former labels)
# Run from: /home/itriu100/carla/

cd pix2pixHD

python test.py \
  --name carla2real_semantic_v6 \
  --dataroot /home/itriu100/carla/datasets/training_semantic_v6 \
  --label_nc 19 \
  --no_instance \
  --loadSize 1024 \
  --fineSize 512 \
  --phase test_Town03 \
  --how_many 1000 \
  --resize_or_crop scale_width_and_crop \
  --gpu_ids 0
