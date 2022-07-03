import os
import logging
import torch

from ..agent import dqn_repr_agent
from ..envs.gridworld import gridworld_envs
from . import networks
from . import laprepr_config_gridworld
from ..tools import flag_tools


class Config(dqn_repr_agent.DqnReprAgentConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        # agent
        flags.batch_size = 128
        flags.discount = 0.98
        flags.update_freq = 50
        flags.update_rate = 0.05
        flags.opt_args.name = 'Adam'
        flags.opt_args.lr = 0.001
        # actor args
        flags.actor_cfg.epsilon_greedy = 0.2
        flags.replay_buffer_init = 10000
        flags.replay_buffer_size = int(1e6)
        flags.replay_update_freq = 1
        flags.replay_update_num = 1
        # train
        flags.log_dir = '/tmp/rl_dqn/log'
        flags.total_train_steps = 200000
        flags.print_freq = 1000
        flags.save_freq = 10000
        flags.test_freq = 1000
        flags.n_test_episodes = 50
        # repr dist reward
        flags.dist_reward_coeff = 1.0
        flags.reward_mode = 'mix'
        flags.repr_model_cfg = flag_tools.Flags(model_ckpt='')

    def _obs_prepro(self, obs):
        return obs.agent.position

    def _goal_obs_prepro(self, obs):
        return obs.goal.position

    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _q_model_factory(self):
        return networks.DiscreteQNetMLP(
                input_shape=self._obs_shape, 
                n_actions=self._action_spec.n, 
                n_layers=3, 
                n_units=256)

    def _repr_model_factory(self):
        model_ckpt = self._flags.repr_model_cfg.model_ckpt
        model_log_dir = os.path.dirname(model_ckpt)
        # load flags when model is trained
        model_flags = flag_tools.load_flags(model_log_dir, 'flags.yaml')
        # check if the model is trained on the same env
        assert model_flags.env_id == self._flags.env_id
        # construct the model config to get the model
        model_config = laprepr_config_gridworld.Config(model_flags)
        model = model_config.args_as_flags.model_cfg.model_factory()
        # load model from checkpoint
        model.load_state_dict(torch.load(model_ckpt))
        logging.info('Representation model loaded from {}.'.format(model_ckpt))
        return model



