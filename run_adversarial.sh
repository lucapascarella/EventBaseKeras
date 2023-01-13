#!/bin/bash

echo "Bash version ${BASH_VERSION}..."

for i in $(seq 0 2 20); do
  python3 "adversarial_steering.py" "--test_dir=data/frames/testing" "--model_path=checkpoints/checkpoints_aps_imgnet_dvsrpt.model" "--frame_mode=aps" "--img_index=$i"
done
