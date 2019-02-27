#!/usr/bin/env bash
DATASET_DIR=./assets/pascalvoc/tfrecord
TRAIN_DIR=./logs/
CHECKPOINT_PATH=./assets/ckpt-mobilenet_v2/mobilenet_v2_1.0_224.ckpt
python ./furiosa_ssd_tf/train_ssd_network_mobilenet_v2.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_0712 \
    --dataset_split_name=trainval \
    --model_name=ssd_512_mobilenet_v2 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --batch_size=32 \
    --checkpoint_exclude_scopes='ssd_512_mobilenet_v2/Extra_feature_layers, ssd_512_mobilenet_v2/Multibox_headers' \
    --checkpoint_model_scope='MobilenetV2' \
    --ignore_missing_vars=True
