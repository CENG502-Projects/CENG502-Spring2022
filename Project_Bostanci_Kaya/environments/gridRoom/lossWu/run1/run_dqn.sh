#!/bin/bash
ENV_ID=$1

python -u -B train_dqn.py \
--env_id=${ENV_ID} \
--log_sub_dir=test \
--args="device='cuda'"