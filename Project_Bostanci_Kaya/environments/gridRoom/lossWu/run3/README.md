# Learning Laplacian Representations in Reinforcement Learning

This codebase implements the representation learning method from [The Laplacian in RL: Learning Representations with Efficient Approximations.](https://openreview.net/forum?id=HJlNpoA5YQ).

The implementation includes (i) representation learning and (ii) using the learned represetations for reward shaping.

This codebase is a re-implementation and was not the one used for generating the experiment results in the paper. The experiment code only includes the grid-world environments but not the Mujoco control ones.

Please refer to `run_full_experiments.sh` for running representation learning, reward shaping, and visualizing representations. `plot_curves.py` is for plotting the learning curve comparisons between different shaped rewards.

The code works with Python>=3.6 and PyTorch>=1.0.

If you use this codebase for your research, please cite the paper:

```
@inproceedings{wu2019laplacian,
  title={The Laplacian in RL: Learning Representations with Efficient Approximations},
  author={Wu, Yifan and Tucker, George and Nachum, Ofir},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## Example Plots

Visualize learned representations:

<p float="left">
  <img src="figures/visualize_reprs/OneRoom.png" width="200" />
  <img src="figures/visualize_reprs/TwoRoom.png" width="200" /> 
  <img src="figures/visualize_reprs/HardMaze.png" width="200" />
</p>

Compare learning curves:

<p float="left">
  <img src="figures/learning_curves/OneRoom.png" width="200" />
  <img src="figures/learning_curves/TwoRoom.png" width="200" /> 
  <img src="figures/learning_curves/HardMaze.png" width="200" />
</p>