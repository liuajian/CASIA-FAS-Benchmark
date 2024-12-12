#!/bin/bash
# custom config

GPU_IDS='0'
DATASET=cdi_data

DATA=/mnt/sdg/CASIA_FasData
OUTPUT=/homedata/ajliu/LAJ/JOBS/xxx

# CASIA_SURF: p1
# CASIA_CeFA: p1.1, p1.3, p1.5, p2.1, p2.2, p4.1, p4.3, p4.5
PROTOCOL=CASIA_CeFA@p1.1
IS_VIDEO=1
TRAINER=CLIP_CDI
IS_FLEXIBLE=0
VERSION=VL              # V or VL
PROMPT=class      # class, engineering, ensembling

CFG=vit_b16             # config file
PREPROCESS=resize_crop_rotate_flip_ColorJitter   ### resize_crop_rotate_flip_ColorJitter

for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}/${VERSION}/${CFG}/${PROTOCOL}/seed${SEED}
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

