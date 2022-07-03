#!/bin/bash
ENV_ID=$1
W_NEG=$2

python -u -B train_laprepr.py \
--env_id=${ENV_ID} \
--log_sub_dir=test \
--args="device='cuda'" \
--args="w_neg=${W_NEG}"