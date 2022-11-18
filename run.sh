#!/bin/bash
# Systematically run the steering predictor on different inputs

echo "Script running..."

FRAME_MODES="aps dvs aps_diff"

EPOCHS=50
BATCH_SIZE=256

TRAIN_DIR="/home/luca/remote/data_50ms/training/"
VAL_DIR="/home/luca/remote/data_50ms/testing/"

COND="True False"

for FRAME_MODE in $FRAME_MODES; do
  for IMGNET in $COND; do
    #    for IMGAUG in $COND; do
    #      for DVSRPT in $COND; do
    CHKPT="checkpoints_$FRAME_MODE"
    if [ "$IMGNET" == "True" ]; then
      CHKPT=$CHKPT"_imgnet"
    fi
    #        if [ "$IMGAUG" == "True" ]; then
    #          CHKPT=$CHKPT"_imgaug"
    #        fi
    #        if [ "$DVSRPT" == "True" ]; then
    #          CHKPT=$CHKPT"_dvsrpt"
    #        fi

    python3 "training.py" "--train_dir=$TRAIN_DIR" "--val_dir=$VAL_DIR" "--checkpoints=$CHKPT" "--frame_mode=$FRAME_MODE" "--use_pretrain=$IMGNET" "--use_augmentation=$IMGAUG" "--dvs_repeat=$DVSRPT" "--batch_size=$BATCH_SIZE" "--epochs=$EPOCHS"
    #      done
    #    done
  done
done
