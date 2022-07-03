from ..agent import laprepr
from ..envs.gridworld import gridworld_envs
from . import networks


class Config(laprepr.LapReprConfig):

    def _set_default_flags(self):
        super()._set_default_flags()
        flags = self._flags
        #flags.d = 20 # Wu's parameter
        flags.d = 10 # Wang's parameter
        #flags.n_samples = 30000 Wu's parameter
        flags.n_samples = 100000 #Wang's parameter
        #flags.batch_size = 128 Wu's parameter
        flags.batch_size = 1024 # Wang's parameter
        flags.discount = 0.9
        flags.w_neg = 1.0
        flags.c_neg = 1.0
        flags.reg_neg = 0.0
        flags.replay_buffer_size = 100000
        flags.opt_args.name = 'Adam'
        flags.opt_args.lr = 0.001
        # train
        flags.log_dir = '/tmp/rl_laprepr/log'
        #flags.total_train_steps = 30000 Wu's parameter
        flags.total_train_steps = 200000 # Wang's parameter
        flags.print_freq = 1000
        flags.save_freq = 10000

    def _obs_prepro(self, obs):
        return obs.agent.position

    def _env_factory(self):
        return gridworld_envs.make(self._flags.env_id)

    def _model_factory(self):
        return networks.ReprNetMLP(
                self._obs_shape, n_layers=3, n_units=256,
                d=self._flags.d)


