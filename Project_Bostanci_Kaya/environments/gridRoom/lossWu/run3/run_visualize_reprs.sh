#!/bin/bash
ENV_ID=$1

python -u -B visualize_reprs.py \
--log_sub_dir=laprepr/${ENV_ID}/test \