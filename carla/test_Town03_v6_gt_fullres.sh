#!/bin/bash
# v6 global network, full 1024x512 inference (scale_width, no crop)
# Run from: /home/itriu100/carla/

cd pix2pixHD

python test.py \
  --name carla2real_semantic_v6 \
  --dataroot /home/itriu100/carla/datasets/training_semantic_v6 \
  --label_nc 19 \
  --no_instance \
  --loadSize 1024 \
  --resize_or_crop scale_width \
  --phase test_Town03_gt \
  --how_many 1000 \
  --results_dir ./results_fullres \
  --gpu_ids 0
