#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <gpu_id> <ratio>"
  exit 1
fi

GPU_ID="$1"
RATIO="$2"

# 推荐写绝对路径
CONFIG_PATH="/p/realai/knowledge-graph/lei/Local_Contribution_Learning/single_gpu.yaml"
TEMP_CONFIG="/p/realai/knowledge-graph/lei/Local_Contribution_Learning/temp_single_gpu.yaml"
LOG_PATH="/p/realai/knowledge-graph/lei/Local_Contribution_Learning/log_folder/${RATIO}.txt"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: $CONFIG_PATH does not exist!"
  exit 2
fi

cp "$CONFIG_PATH" "$TEMP_CONFIG"
if [ ! -f "$TEMP_CONFIG" ]; then
  echo "Error: failed to create temp config $TEMP_CONFIG"
  exit 3
fi

sed -i "s/^gpu_ids: .*/gpu_ids: '${GPU_ID}'/" "$TEMP_CONFIG"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

nohup accelerate launch --config_file "$TEMP_CONFIG" experiment_acce.py \
    --batch_size 16 --gaussian_num 8 --prototype_num 2 \
    --data_set out_hospital_mortality_30 --ratio "${RATIO}" \
    --bert_model_name Simonlee711/Clinical_ModernBERT --max_length 8192 \
    > "${LOG_PATH}" &

# rm "$TEMP_CONFIG"