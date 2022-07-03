#!/bin/bash
ENV_ID=$1
REWARD_MODE=$2

python -u -B train_dqn_repr.py \
--env_id=${ENV_ID} \
--log_sub_dir=${REWARD_MODE} \
--repr_ckpt_sub_path=laprepr/${ENV_ID}/test/model.ckpt \
--reward_mode=${REWARD_MODE} \
--args="device='cuda'"