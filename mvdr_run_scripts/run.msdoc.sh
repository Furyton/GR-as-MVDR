#!/bin/bash

COMMON_PARAMS_FILE=common.params.msdoc.320k.sh
source common_settings/$COMMON_PARAMS_FILE

MODEL_PATH="/path/to/t5-v1_1-base"

PIPELINE=train
CHECKPOINT=

N_GPU=1

SIMILARITY_METRICS=cosine
NPROBE=10
FAISS_DEPTH=$TOPK

DIM=128

NUM_WORKERS=8

IN_BATCH_NEG=true
TRAIN_BS=128
TRAIN_ACC=2
TRAIN_N_EPOCH=5
TRAIN_LR=5e-4 # 3e-6
TRAIN_GRAD_CLIP=2.0
TRAIN_SAVE_STEP=0
TRAIN_TOT_N_CKPTS=1

INDEX_BS=1024
RANK_BS=512
# experiment dirs

PARAMS=mvdr.$(basename $MODEL_PATH).$PIPELINE.bs$(($TRAIN_BS*$TRAIN_ACC)).epoch$TRAIN_N_EPOCH.lr$TRAIN_LR.clip.$TRAIN_GRAD_CLIP

if $IN_BATCH_NEG; then
    PARAMS+=.batch_neg$TRAIN_BS
fi

TIMESTAMP=$(date -d "today" +"%m%d.%H.%M")

# PARAMS+=$TIMESTAMP
PROJECT=mvdr.$DATA_PARAMS.$PARAMS
EXP_DIR=$OUTPUT_ROOT/$PARAMS
LOG=$EXP_DIR/log
LOGFILE=$LOG/mvdr.$TIMESTAMP.log

# setting env

mkdir -p $EXP_DIR
mkdir -p $LOG

cp common_settings/$COMMON_PARAMS_FILE $LOG/$COMMON_PARAMS_FILE
cp mvdr_run_scripts/run.nq.sh $LOG/run.nq.sh
cp mvdr.py $LOG/mvdr.py

export OMP_NUM_THREADS=$(($NUM_WORKERS * 2))

python mvdr.py \
    --pipeline $PIPELINE \
    --project $PROJECT \
    --model_name_or_path $MODEL_PATH \
    --triples $TRIPLES \
    --queries $QUERIES \
    --collection $COLLECTIONS \
    --qrels $QRELS \
    --output_dir $EXP_DIR \
    --data_name $(basename $DATA_BASE) \
    --query_maxlen $Q_MAXLEN \
    --doc_maxlen $D_MAXLEN \
    --dim $DIM \
    --similarity_metric $SIMILARITY_METRICS \
    --train_batch_size $TRAIN_BS \
    --index_batch_size $INDEX_BS \
    --retrieval_batch_size $RANK_BS \
    --accumsteps $TRAIN_ACC \
    --n_epochs $TRAIN_N_EPOCH \
    --lr $TRAIN_LR \
    --max_grad_norm $TRAIN_GRAD_CLIP \
    --save_steps $TRAIN_SAVE_STEP \
    --tot_n_ckpts $TRAIN_TOT_N_CKPTS \
    --num_workers $NUM_WORKERS \
    --sample_fraction 0.2 \
    --nprobe $NPROBE \
    --faiss_depth $FAISS_DEPTH \
    --at $AT \
    $(if $IN_BATCH_NEG; then echo "--in_batch_negative"; fi) \
    $(if [ ! -z $CHECKPOINT ]; then echo "--checkpoint $CHECKPOINT"; fi) \
    $(if $RM_STOPWORDS; then echo "--rm_stopwords"; fi) \
    $(if $RM_PUNCTUATION; then echo "--rm_punctuation"; fi) \
    $(if $LOWER; then echo "--lower"; fi) \
    2>&1 | tee $LOGFILE
