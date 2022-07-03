from ..agent import dqn_agent
from ..envs.gridworld import gridworld_envs
from . import networks


class Config(dqn_agent.DqnAgentConfig):

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

    def _obs_prepro(self, obs):
        return obs.agent.position

    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _q_model_factory(self):
        return networks.DiscreteQNetMLP(
                input_shape=self._obs_shape, 
                n_actions=self._action_spec.n, 
                n_layers=3, 
                n_units=256)


