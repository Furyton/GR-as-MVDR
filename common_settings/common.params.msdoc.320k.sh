#!/bin/bash

set -euxo pipefail

export DATA_BASE="/path/to/msdoc320k"

export QRELS=$DATA_BASE/qrel.dev.tsv
export TRIPLES=$DATA_BASE/triples.train.tsv
export COLLECTIONS=$DATA_BASE/collection.tsv
export QUERIES=$DATA_BASE/queries.all.tsv
export EVAL_QUERIES=$DATA_BASE/queries.dev.tsv
export CANDIDATES=bm25.msdoc320k/candidate.json
export BM25_INDEX=bm25.msdoc320k.index

export Q_MAXLEN=64
export D_MAXLEN=256

export RM_STOPWORDS=false
export RM_PUNCTUATION=true
export LOWER=false

export TOPK=10

# eval
export AT="1 5 10"

# output

DATA_PARAMS=$(basename $DATA_BASE).q$Q_MAXLEN.d$D_MAXLEN
if $RM_STOPWORDS; then
    DATA_PARAMS+=.rs
fi
if $RM_PUNCTUATION; then
    DATA_PARAMS+=.rp
fi
if $LOWER; then
    DATA_PARAMS+=.low
fi
export DATA_PARAMS=$DATA_PARAMS

export OUTPUT_ROOT=output.$DATA_PARAMS

mkdir -p $OUTPUT_ROOT

