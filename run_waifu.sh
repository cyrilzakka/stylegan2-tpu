#!/bin/bash
set -ex
. discord_token.sh
export TPU_HOST=10.255.128.3
export TPU_NAME=${TPU_NAME:-tpu-v3-32-euw4a-2}
export RESOLUTION=${RESOLUTION:-512}
export NUM_CHANNELS=3
export LABEL_SIZE=128
export MODEL_DIR=${1:-'gs://arfa-euw4a/results/danbooru-512-conditional-subset-128/run3'}
export COUNT=${COUNT:-1}
export TIMEOUT_IN_MS=${TIMEOUT_IN_MS:-$((3*60*1000))}
exec python3 waifu.py
