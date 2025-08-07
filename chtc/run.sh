#!/bin/bash
set -e
echo "Running with args : $@"
for ((i=1; i<=$#; i++)); do
  if [[ ${!i} == "--shard_path" ]]; then
    next=$((i+1))
    SHARD_FILE=${!next}             
    SHARD_ID=$(echo "$SHARD_FILE" | grep -oP '\d+(?=\.tar)')  
    break
  fi
done

export DATA_PATH=$(pwd)

export MPLCONFIGDIR=${DATA_PATH}/matplotlib
export TRANSFORMERS_CACHE=${DATA_PATH}/hf_cache
export TORCH_HOME=/staging/zzhu362/torch_cache
export HF_HOME=${DATA_PATH}/hf_home
export XDG_CACHE_HOME=${DATA_PATH}/xdg_cache

cd /staging/zzhu362/facial_features_3d

source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch3d

python main_mer_npy.py "$@"

cd /staging/zzhu362/mer_output

tar -czf mer2025_${SHARD_ID}.tar.gz -C "$DATA_PATH/openFace_emoca" .
