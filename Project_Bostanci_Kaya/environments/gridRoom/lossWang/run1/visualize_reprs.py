"""Visualize learned representation."""
import os
import argparse
import importlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import torch as th
import pandas as pd


import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

from rl_lap.agent import laprepr
from rl_lap.tools import flag_tools
from rl_lap.tools import torch_tools


parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_sub_dir', type=str, 
        default='laprepr/OneRoom/test')
parser.add_argument('--output_sub_dir', type=str, 
        default='visualize_reprs')
parser.add_argument('--config_dir', type=str, default='rl_lap.configs')
parser.add_argument('--config_file', 
        type=str, default='laprepr_config_gridworld')


FLAGS = parser.parse_args()


def get_config_cls():
    config_module = importlib.import_module(
            FLAGS.config_dir+'.'+FLAGS.config_file)
    config_cls = config_module.Config
    return config_cls


# Added by Semih
def get_r_states(positionEmbedding, env):
    r_states = th.full((env.task.maze.width, env.task.maze.height), fill_value=-1, dtype=th.long)
    r_states[positionEmbedding[:, 0], positionEmbedding[:, 1]] = th.arange(0, len(positionEmbedding))
    return r_states


def get_exact_laplacian(states, r_states, n_actions=4):
    # Build adjacency matrix
    A = th.zeros((len(states), len(states)))
    states_map = r_states > -1
    cur_pos = r_states[states_map]
    left_pos = r_states[states_map.roll(shifts=-1, dims=1)]
    left_pos[left_pos == -1] = cur_pos[left_pos == -1]
    A[cur_pos, left_pos] = 1
    right_pos = r_states[states_map.roll(shifts=1, dims=1)]
    right_pos[right_pos == -1] = cur_pos[right_pos == -1]
    A[cur_pos, right_pos] = 1
    up_pos = r_states[states_map.roll(shifts=-1, dims=0)]
    up_pos[up_pos == -1] = cur_pos[up_pos == -1]
    A[cur_pos, up_pos] = 1
    down_pos = r_states[states_map.roll(shifts=1, dims=0)]
    down_pos[down_pos == -1] = cur_pos[down_pos == -1]
    A[cur_pos, down_pos] = 1

    # Build transition matrix
    P = A / n_actions
    P[range(len(states)), range(len(states))] += 1 - P.sum(axis=0)

    # Compute graph Laplacian
    D = A.sum(axis=-1)
    L = th.diag(D) - A

    # Eigendecompose the Laplacian
    eigenvalues, eigenvectors = th.eig(L, eigenvectors=True)
    idx = eigenvalues[:, 0].argsort()
    eigenvalues = eigenvalues[idx, 0]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors, eigenvalues, P

def main():
    # setup log directories
    log_dir = os.path.join(FLAGS.log_base_dir, FLAGS.log_sub_dir)
    output_dir = os.path.join(FLAGS.log_base_dir, FLAGS.output_sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # load config
    flags = flag_tools.load_flags(log_dir)
    cfg_cls = get_config_cls()
    cfg = cfg_cls(flags)
    learner_args = cfg.args_as_flags
    device = learner_args.device
    # load model from checkpoint
    model = learner_args.model_cfg.model_factory()
    model.to(device=device)
    ckpt_path = os.path.join(log_dir, 'model.ckpt')
    model.load_state_dict(torch.load(ckpt_path))
    # -- use loaded model to get state representations --
    # get the full batch of states from env
    env = learner_args.env_factory()
    obs_prepro = learner_args.obs_prepro
    n_states = env.task.maze.n_states
    pos_batch = env.task.maze.all_empty_grids()
    obs_batch = [env.task.pos_to_obs(pos_batch[i]) for i in range(n_states)]
    states_batch = np.array([obs_prepro(obs) for obs in obs_batch])
    # get goal state representation
    goal_pos = env.task.goal_pos
    goal_obs = env.task.pos_to_obs(goal_pos)
    goal_state = obs_prepro(goal_obs)[None]
    # get representations from loaded model
    states_torch = torch_tools.to_tensor(states_batch, device)
    goal_torch = torch_tools.to_tensor(goal_state, device)
    states_reprs = model(states_torch).detach().cpu().numpy()
    goal_repr = model(goal_torch).detach().cpu().numpy()
    # compute l2 distances from states to goal

    ##### Wu's implementation for visualization
    #l2_dists = np.sqrt(np.sum(np.square(states_reprs - goal_repr), axis=-1))
    ##### Wu's implementation for visualization

    directory = "csvResults"
    if os.path.isfile(directory) == 0:
        try:
            os.makedirs(directory, exist_ok = True)
            print("Directory '%s' created successfully" % directory)
        except OSError as error:
            print("Directory '%s' can not be created" % directory)

    # Save states_torch variable
    t_np = states_torch.numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv("csvResults/statesfile.csv",index=False) #save to file

    # Get reverse state representations
    r_states_torch = get_r_states(pos_batch, env)
    # Find the eigenvectors of the graph Laplacian
    [groundTruthEigenvectors, groundTruthEigenvalues, stateTransitionMatrix] = get_exact_laplacian(states_torch, r_states_torch, n_actions=4)
    
    # Save r_states_torch variable
    t_np = r_states_torch.numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv("csvResults/rStatesfile.csv",index=False) #save to file

    # Save groundTruthEigenvalues
    t_np = groundTruthEigenvalues[: states_reprs.shape[1]].numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv("csvResults/groundTruthEigenvalues.csv",index=False) #save to file

    # Save groundTruthEigenvectors
    t_np = groundTruthEigenvectors[:, : states_reprs.shape[1]].numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv("csvResults/groundTruthEigenvectors.csv",index=False) #save to file

    # Save state representations
    df = pd.DataFrame(states_reprs) #convert to a dataframe
    df.to_csv("csvResults/stateRepresentations.csv",index=False) #save to file

    # -- visialize state representations --
    # plot raw distances with the walls
    image_shape = goal_obs.agent.image.shape
    map_ = np.zeros(image_shape[:2], dtype=np.float32)
    ##### Wu's implementation for visualization
    #map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
    ### Wang's implementation  -- l2distance values  are substituted with state reperesentations
    for dimensionIndex in range(states_reprs.shape[1]): 
        map_[pos_batch[:, 0], pos_batch[:, 1]] = -states_reprs[:, dimensionIndex]/np.linalg.norm(states_reprs[:, dimensionIndex])
        im_ = plt.imshow(map_, interpolation='none', cmap='bwr')
        plt.colorbar()
        # add the walls to the normalized distance plot
        walls = np.expand_dims(env.task.maze.render(), axis=-1)
        map_2 = im_.cmap(im_.norm(map_))
        #map_2 = im_.cmap(map_)
        map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
        map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
        #map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
        plt.cla()
        plt.imshow(map_2, interpolation='none')
        plt.xticks([])
        plt.yticks([])
        figfile = os.path.join(output_dir, '{}'.format(flags.env_id))
        suffixString = "appr_dimension" + str(dimensionIndex+1) + ".pdf"
        figfile = figfile + suffixString
        plt.savefig(figfile, bbox_inches='tight')
        plt.clf()

    map_ = np.zeros(image_shape[:2], dtype=np.float32)
    ##### Wu's implementation for visualization of ground truth 
    #map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
    ### Wang's implementation  -- l2distance values  are substituted with state reperesentations
    for dimensionIndex in range(states_reprs.shape[1]): 
        map_[pos_batch[:, 0], pos_batch[:, 1]] = groundTruthEigenvectors[:, dimensionIndex]
        im_ = plt.imshow(map_, interpolation='none', cmap='bwr')
        plt.colorbar()
        # add the walls to the normalized distance plot
        walls = np.expand_dims(env.task.maze.render(), axis=-1)
        map_2 = im_.cmap(im_.norm(map_))
        #map_2 = im_.cmap(map_)
        map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
        map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
        #map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
        plt.cla()
        plt.imshow(map_2, interpolation='none')
        plt.xticks([])
        plt.yticks([])
        figfile = os.path.join(output_dir, '{}'.format(flags.env_id))
        suffixString = "gt_dimension" + str(dimensionIndex+1) + ".pdf"
        figfile = figfile + suffixString
        plt.savefig(figfile, bbox_inches='tight')
        plt.clf()


    map_ = np.zeros(image_shape[:2], dtype=np.float32)
    ##### Wu's implementation for visualization of ground truth 
    #map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
    ### Wang's implementation  -- l2distance values  are substituted with state reperesentations
    for dimensionIndex in range(states_reprs.shape[1]): 
        map_[pos_batch[:, 0], pos_batch[:, 1]] = groundTruthEigenvectors[:, dimensionIndex]
        plt.subplot(1,10,dimensionIndex+1)
        im_ = plt.imshow(map_, interpolation='none', cmap='bwr')
        plt.colorbar()
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        plt.gca()
        # add the walls to the normalized distance plot
        walls = np.expand_dims(env.task.maze.render(), axis=-1)
        map_2 = im_.cmap(im_.norm(map_))
        #map_2 = im_.cmap(map_)
        map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
        map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
        #map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
        #plt.cla()
        plt.imshow(map_2, interpolation='none')
        plt.xticks([])
        plt.yticks([])
    figfile = os.path.join(output_dir, '{}'.format(flags.env_id))
    suffixString = "gt_alldim.pdf"
    figfile = figfile + suffixString
    # plt.savefig(figfile, bbox_inches='tight')
    plt.savefig(figfile)
    plt.clf()

if __name__ == '__main__':
    main()

