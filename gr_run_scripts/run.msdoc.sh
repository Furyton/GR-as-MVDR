#!/bin/bash

COMMON_PARAMS_FILE=common.params.msdoc.320k.sh
MODEL_PATH="/path/to/t5-v1_1-base"

source common_settings/$COMMON_PARAMS_FILE

PIPELINE=train
CHECKPOINT=

N_GPU=1

USE_SPAN=true
N_SPAN=10
SPAN_LENGTH=10
NGRAM=3
TEMPERATURE=1.0

NUM_WORKERS=16

TRAIN_BS=64
TRAIN_ACC=4
TRAIN_N_EPOCH=3
# TRAIN_LR=1e-4 # base
TRAIN_LR=1e-4 # v1_1
TRAIN_GRAD_CLIP=2.0
TRAIN_SAVE_STEP=0
TRAIN_TOT_N_CKPTS=1

RANK_BS=256

TOPK=$TOPK

# experiment dirs

TIMESTAMP=$(date -d "today" +"%m%d.%H.%M")

PARAMS=gr.noquery_aug.$(basename $MODEL_PATH).$PIPELINE.bs$(($TRAIN_BS*$TRAIN_ACC)).epoch$TRAIN_N_EPOCH.lr$TRAIN_LR.clip.$TRAIN_GRAD_CLIP.

PROJECT=gr.$DATA_PARAMS.$PARAMS
EXP_DIR=$OUTPUT_ROOT/$PARAMS
LOG=$EXP_DIR/log
LOGFILE=$LOG/gr.$TIMESTAMP.log

# setting env

mkdir -p $EXP_DIR
mkdir -p $LOG

cp common_settings/$COMMON_PARAMS_FILE $LOG/$COMMON_PARAMS_FILE
cp gr_run_scripts/run.nq.sh $LOG/run.nq.sh
cp gr.py $LOG/gr.py

export OMP_NUM_THREADS=$(($NUM_WORKERS * 2))

#accelerate launch --num_processes $N_GPU gr.py \
python gr.py \
    --project $PROJECT \
    --pipeline $PIPELINE \
    --model_name_or_path $MODEL_PATH \
    --triples $TRIPLES \
    --queries $QUERIES \
    --collection $COLLECTIONS \
    --qrels $QRELS \
    --output_dir $EXP_DIR \
    --data_name $(basename $DATA_BASE) \
    --query_maxlen $Q_MAXLEN \
    --doc_maxlen $D_MAXLEN \
    --train_batch_size $TRAIN_BS \
    --retrieval_batch_size $RANK_BS \
    --accumsteps $TRAIN_ACC \
    --n_epochs $TRAIN_N_EPOCH \
    --lr $TRAIN_LR \
    --max_grad_norm $TRAIN_GRAD_CLIP \
    --save_steps $TRAIN_SAVE_STEP \
    --tot_n_ckpts $TRAIN_TOT_N_CKPTS \
    --num_workers $NUM_WORKERS \
    --topk $TOPK \
    --at $AT \
    $(if [ ! -z $CHECKPOINT ]; then echo "--checkpoint $CHECKPOINT"; fi) \
    $(if $USE_SPAN; then echo "--use_span --n_span $N_SPAN --span_length $SPAN_LENGTH --ngram $NGRAM --temperature $TEMPERATURE"; fi) \
    $(if $RM_STOPWORDS; then echo "--rm_stopwords"; fi) \
    $(if $RM_PUNCTUATION; then echo "--rm_punctuation"; fi) \
    $(if $LOWER; then echo "--lower"; fi) \
    2>&1 | tee $LOGFILE
