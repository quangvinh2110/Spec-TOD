#!/bin/bash
set -ex


MODEL=/home/LFew-TOD/ckpt/lora-out-axolotl/Llama-3-pattent-loss-all/checkpoint-5505/merged
MAX_SEQ_LEN=4096
MAX_INPUT_LEN=4000
DEVICE=0 # 1,2,3,4,5,6,7
NAME=tgi_${DEVICE//,/_}_1
PORT=8030

docker run \
    --privileged --security-opt "seccomp=unconfined" \
    -e CUDA_VISIBLE_DEVICES=${DEVICE} \
    --gpus all \
    --shm-size 1g \
    -p ${PORT}:80 -d \
    --name $NAME \
    -v tgi-data:/data \
    -v /home:/home \
    ghcr.io/huggingface/text-generation-inference:2.2.0 \
    --model-id $MODEL \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-input-length $MAX_INPUT_LEN \
    --max-total-tokens $MAX_SEQ_LEN \
    --max-batch-prefill-tokens $MAX_SEQ_LEN \

docker logs -f ${NAME}

# 61eb4dd71f7d
