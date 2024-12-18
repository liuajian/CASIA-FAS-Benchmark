#!/bin/bash
# custom config

GPU_IDS='7'
DATASET=c_data

DATA=/mnt/sdg/CASIA_FasData
OUTPUT=/homedata/ajliu/LAJ/JOBS/UniAttackData-VLengineering

# CASIA_HiFiMask:   p1, p2.1, p2.2, p2.3, p3
# CASIA_SuHiFiMask: p1, p2.1, p2.2, p2.3, p2.4, p3
# UniAttackData:    p1, p2.1, p2.2
PROTOCOL=UniAttackData@p1
IS_VIDEO=0
TRAINER=CLIP
VERSION=VL              # V or VL
PROMPT=engineering      # class, engineering, ensembling

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

