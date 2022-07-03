"""Visualize learned representation."""
import os
import argparse
import importlib

import numpy as np
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


parser = argparse.ArgumentParser()
parser.add_argument('--log_base_dir', type=str, 
        default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--output_sub_dir', type=str, 
        default='learning_curves')


FLAGS = parser.parse_args()




def main():
    # setup log directories
    output_dir = os.path.join(FLAGS.log_base_dir, FLAGS.output_sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    envs = ['OneRoom', 'TwoRoom', 'HardMaze']
    r_modes = ['sparse', 'mix', 'l2', 'rawmix']
    colors = ['royalblue', 'darkorange', 'seagreen', 'tomato']
    linestyles = ['--', '-', '-.', ':']
    linewidth = 3
    for env_id in envs:
        loaded_results = {}
        for r_mode in r_modes:
            log_dir = os.path.join(
                    FLAGS.log_base_dir, 'dqn_repr', env_id, r_mode)
            results_file = os.path.join(log_dir, 'results.csv')
            results = np.loadtxt(results_file, delimiter=',')
            loaded_results[r_mode] = results
        # plot
        handles = []
        for r_mode, c, ls in zip(r_modes, colors, linestyles):
            x = loaded_results[r_mode][:, 0]
            y = loaded_results[r_mode][:, 1]
            h, = plt.plot(x, y, color=c, linestyle=ls, linewidth=linewidth,
                    label=r_mode)
            handles.append(h)
        plt.title(env_id)
        plt.legend(handles=handles)
        plt.xlabel('train steps')
        plt.ylabel('episodic returns')
        figfile = os.path.join(output_dir, '{}.png'.format(env_id))
        plt.savefig(figfile, bbox_inches='tight')
        plt.clf()
        print('Plot saved at {}.'.format(figfile))


if __name__ == '__main__':
    main()

