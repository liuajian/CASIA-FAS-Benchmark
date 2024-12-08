#!/bin/bash
# custom config

GPU_IDS='0'
DATASET=cdi_data

DATA=/mnt/sdg/CASIA_FasData
OUTPUT=/homedata/ajliu/LAJ/JOBS/list
PROTOCOL=CASIA_SURF@p1
IS_VIDEO=1
TRAINER=CLIP_CDI
IS_FLEXIBLE=1
VERSION=V              # V or VL
PROMPT=class      # class, engineering, ensembling

CFG=vit_b16             # config file
PREPROCESS=resize_crop_rotate_flip_ColorJitter   ### resize_crop_rotate_flip_ColorJitter

for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}/${VERSION}@${IS_VIDEO}@${IS_FLEXIBLE}/${CFG}/${PROTOCOL}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --gpu_ids ${GPU_IDS} \
    --root ${DATA} \
    --protocol ${PROTOCOL} \
    --is_video ${IS_VIDEO} \
    --is_flexible ${IS_FLEXIBLE} \
    --preprocess ${PREPROCESS} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --prompt ${PROMPT} \
    --version ${VERSION}
    # fi
done

