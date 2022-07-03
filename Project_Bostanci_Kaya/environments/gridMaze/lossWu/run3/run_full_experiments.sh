#!/bin/bash

# learn representations
./run_laprepr.sh OneRoom 5.0 && \
./run_laprepr.sh TwoRoom 5.0 && \
./run_laprepr.sh HardMaze 1.0

# visualize representations
./run_visualize_reprs.sh OneRoom && \
./run_visualize_reprs.sh TwoRoom && \
./run_visualize_reprs.sh HardMaze

# train agent with shaped rewards
for r_mode in mix rawmix l2 sparse
do
    ./run_dqn_repr.sh OneRoom ${r_mode} && \
    ./run_dqn_repr.sh TwoRoom ${r_mode} && \
    ./run_dqn_repr.sh HardMaze ${r_mode}
done