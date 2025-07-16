#!/bin/bash

DATASET_DIR="/home/works/dtu"   # path to dataset folder
LOGDIR="outputs/DIV-MVS-2"
MASTER_ADDR="localhost"
MASTER_PORT=1234
NNODES=1
NGPUS=1
NODE_RANK=0

# Ensure log directory exists
mkdir -p $LOGDIR

# Make all 8 GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: Enable NCCL debug logging for troubleshooting
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT

# Optional: Set OpenMP threads to 1 per process (recommended for DDP)
export OMP_NUM_THREADS=1

# Launch with torchrun
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NGPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
    --logdir $LOGDIR \
    --dataset custom_train \
    --trainpath /home/works/coolant-dataset/dataset \
    --trainlist lists/coolant/train.txt \
    --testpath $DATASET_DIR \
    --ngroups 8,4,2 \
    --batch_size 2 \
    --lr 0.0005 | tee -a $LOGDIR/log.txt
