#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=750
CURR_QUERY=150

# TEST
CUDA_VISIBLE_DEVICES=3 python main_instance_segmentation.py \
general.checkpoint="checkpoint/replica.ckpt" \
data/datasets=replica \
general.train_mode=false \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}