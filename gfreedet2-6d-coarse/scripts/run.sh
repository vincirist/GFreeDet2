#!/usr/bin/env bash
set -x
this_dir=$(dirname "$0")
# commonly used opts:


SCRIPT=$1
CFG=$2
GPU_DEVICES=$3
IFS=',' read -ra GPUS <<< "$GPU_DEVICES"
# GPUS=($(echo "$GPU_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $GPU_DEVICES num gpus: $NGPU"
# CUDA_LAUNCH_BLOCKING=1
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONPATH="$this_dir/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$GPU_DEVICES CFG_FILE=$CFG \
python $SCRIPT \
    ${@:4}