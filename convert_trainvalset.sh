#!/bin/bash
DATASET_DIR=./assets/pascalvoc/VOCDevkit/VOC07+12/
OUTPUT_DIR=./assets/pascalvoc/tfrecord
python ./furiosa_ssd_tf/tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_07+12_trainval \
    --output_dir=${OUTPUT_DIR}
