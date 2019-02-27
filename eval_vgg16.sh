#!/usr/bin/env bash
DATASET_DIR=./assets/pascalvoc/tfrecord
EVAL_DIR=./logs/eval
CHECKPOINT_PATH=./logs/
python ./furiosa_ssd_tf/eval_ssd_network_mobilenet_v2.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_512_mobilenet_v2 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1