#!/bin/bash
# custom config

GPU_IDS='5'
DATASET=dg_data

DATA=/mnt/sdh/LAJ_data/CASIA_FasData
OUTPUT=/homedata/ajliu/LAJ/JOBS/icmo_cspl

# ICMO: CASIA_FASD-MSU_MFSD-ReplayAttack@OULU_NPU
# SCW:  CASIA_CeFA-CASIA_SURF@IDIAP_WMCA
# DKMH: CASIA_SuHiFiMask-3DMask-CASIA_HiFiMask@HKBUv2
PROTOCOL=CASIA_FASD-OULU_NPU-ReplayAttack@MSU_MFSD
TRAINER=ICPE

CFG=vit_b16             # config file
PREPROCESS=resize_crop_rotate_flip_ColorJitter   ### resize_crop_rotate_flip_ColorJitter

for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}/${CFG}/${PROTOCOL}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --gpu_ids ${GPU_IDS} \
    --root ${DATA} \
    --protocol ${PROTOCOL} \
    --preprocess ${PREPROCESS} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    # fi
done

for SEED in 1
do
    DIR=${OUTPUT}/${TRAINER}/${CFG}/${PROTOCOL}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --gpu_ids ${GPU_IDS} \
    --root ${DATA} \
    --protocol ${PROTOCOL} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir ${DIR} \
    --output-dir ${DIR} \
    --eval-only
    # fi
done
