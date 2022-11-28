#!/bin/bash
# Systematically run the steering predictor on different inputs

echo "Script running..."

FRAME_MODES="aps dvs"

EPOCHS=50
BATCH_SIZE=64

TRAIN_DIR="frames/training/"
VAL_DIR="frames/testing/"

COND="False True"

for FRAME_MODE in $FRAME_MODES; do
  for IMGNET in $COND; do
    for IMGAUG in $COND; do
      for DVSRPT in $COND; do
      CHKPT="checkpoints_$FRAME_MODE"
      if [ "$IMGNET" == "True" ]; then
        CHKPT=$CHKPT"_imgnet"
      fi
      if [ "$IMGAUG" == "True" ]; then
        CHKPT=$CHKPT"_imgaug"
      fi
      if [ "$DVSRPT" == "True" ]; then
        CHKPT=$CHKPT"_dvsrpt"
      fi

#      if [ "$IMGAUG" != "False" ] && [ "$DVSRPT" != "False" ] ; then
        echo "training.py" "--train_dir=$TRAIN_DIR" "--val_dir=$VAL_DIR" "--checkpoints=$CHKPT" "--frame_mode=$FRAME_MODE" "--use_pretrain=$IMGNET" "--use_augmentation=$IMGAUG" "--dvs_repeat=$DVSRPT" "--batch_size=$BATCH_SIZE" "--epochs=$EPOCHS"
#        python3 "training.py" "--train_dir=$TRAIN_DIR" "--val_dir=$VAL_DIR" "--checkpoints=$CHKPT" "--frame_mode=$FRAME_MODE" "--use_pretrain=$IMGNET" "--use_augmentation=$IMGAUG" "--dvs_repeat=$DVSRPT" "--batch_size=$BATCH_SIZE" "--epochs=$EPOCHS"
#      fi
      done
    done
  done
done