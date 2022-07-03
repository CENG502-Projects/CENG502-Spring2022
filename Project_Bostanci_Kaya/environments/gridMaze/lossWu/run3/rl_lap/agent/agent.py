import os
import logging
import collections

import numpy as np
import torch

from . import episodic_replay_buffer
from ..envs import actors
from ..envs import evaluator as evaluator_lib

from ..tools import py_tools
from ..tools import torch_tools
from ..tools import flag_tools
from ..tools import summary_tools
from ..tools import timer_tools


class Agent:

    @py_tools.store_args
    def __init__(self,
            # pytorch
            device=None,
            # env args
            action_spec=None,
            obs_shape=None,
            obs_prepro=None,
            env_factory=None,
            # model args
            model_cfg=None,
            optimizer_cfg=None,
            batch_size=128,
            discount=0.99,
            update_freq=1,
            update_rate=0.001,
            # actor args
            actor_cfg=None,  # e.g. exploration
            replay_buffer_init=10000,
            replay_buffer_size=int(1e6),
            replay_update_freq=1,
            replay_update_num=1,
            # training args
            log_dir='/tmp/rl/log',
            total_train_steps=50000,
            print_freq=1000,
            save_freq=10000,
            test_freq=5000,
            n_test_episodes=50,
            ):
        self._build_agent()


    def _build_agent(self):
        if self._device is None:
            self._device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('device: {}.'.format(self._device))
        self._build_model()
        self._build_optimizer()
        self._replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
                max_size=self._replay_buffer_size)
        self._global_step = 0
        self._train_info = collections.OrderedDict()

    def _build_optimizer(self): 
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _build_loss(self, batch):
        raise NotImplementedError
    
    def _update_target_vars(self):
        # requires self._vars_learning and self._vars_target as state_dict`s
        for var_name, var_t in self._vars_target.items():
            updated_val = (self._update_rate
                    * self._vars_learning[var_name].data
                    + (1.0 - self._update_rate) * var_t.data)
            var_t.data.copy_(updated_val)

    def _train_policy_fn(self, state):
        raise NotImplementedError
        # return action, context

    def _test_policy_fn(self, state):
        raise NotImplementedError
        # return action, context

    def _random_policy_fn(self, state):
        return self._action_spec.sample(), None
        # return action, context

    def _get_obs_batch(self, steps):
        obs_batch = [self._obs_prepro(s.step.time_step.observation)
                for s in steps]
        return np.stack(obs_batch, axis=0)

    def _get_action_batch(self, steps):
        action_batch = [s.step.action for s in steps]
        return np.stack(action_batch, axis=0)

    def _get_r_dsc_batch(self, steps2):
        """
        Compute discount based on s_t+1, discount is 0 when
        s_t+1 is the final state, self._discount otherwise.
        """
        is_final_s2 = np.array([s.step.time_step.is_final for s in steps2]
                ).astype(np.float32)
        dsc = (1.0 - is_final_s2) * self._discount
        r = np.array([s.step.time_step.reward for s in steps2])
        return r, dsc

    def _tensor(self, x):
        return torch_tools.to_tensor(x, self._device)

    def _get_train_batch(self):
        steps1, steps2 = self._replay_buffer.sample_transitions(
                batch_size=self._batch_size)
        a = self._get_action_batch(steps1)
        s1, s2 = map(self._get_obs_batch, [steps1, steps2])
        # compute reward and discount
        r, dsc = self._get_r_dsc_batch(steps2)
        batch = flag_tools.Flags()
        batch.s1 = self._tensor(s1)
        batch.s2 = self._tensor(s2)
        batch.r = self._tensor(r)
        batch.dsc = self._tensor(dsc)
        batch.a = self._tensor(a)
        return batch

    def _train_step(self):
        train_batch = self._get_train_batch()
        loss = self._build_loss(train_batch)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._global_step += 1
        if self._global_step % self._update_freq == 0:
            self._update_target_vars()

    def _print_train_info(self):
        summary_str = summary_tools.get_summary_str(
                step=self._global_step, info=self._train_info)
        logging.info(summary_str)

    def train(self):
        log_dir = self._log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        actor = actors.StepActor(self._env_factory)
        evaluator = evaluator_lib.Evaluator(self._env_factory)
        results_file = os.path.join(log_dir, 'results.csv')
        # start actors, collect trajectories from random actions
        logging.info('Start collecting transitions.')
        train_timer = timer_tools.Timer()
        # collect initial transitions
        total_n_steps = 0
        collect_batch = 10000
        while total_n_steps < self._replay_buffer_init:
            n_steps = min(collect_batch, 
                    self._replay_buffer_init - total_n_steps)
            steps = actor.get_steps(n_steps, self._random_policy_fn)
            self._replay_buffer.add_steps(steps)
            total_n_steps += n_steps
            logging.info('({}/{}) steps collected.'
                .format(total_n_steps, self._replay_buffer_init))
        time_cost = train_timer.time_cost()
        logging.info('Replay buffer initialization finished, time cost: {}s'
            .format(time_cost))
        # learning begins
        train_timer.set_step(0)
        test_results = []
        for step in range(self._total_train_steps):
            assert step == self._global_step
            self._train_step()
            # update replay buffer:
            if (step + 1) % self._replay_update_freq == 0:
                steps = actor.get_steps(self._replay_update_num,
                        self._train_policy_fn)
            self._replay_buffer.add_steps(steps)
            # save
            if (step + 1) % self._save_freq == 0:
                saver_path = os.path.join(log_dir, 
                        'agent-{}.ckpt'.format(step+1))
                self.save_ckpt(saver_path)
            # print info
            if step == 0 or (step + 1) % self._print_freq == 0:
                steps_per_sec = train_timer.steps_per_sec(step)
                logging.info('Training steps per second: {:.4g}.'
                        .format(steps_per_sec))
                self._print_train_info()
            # test
            if step == 0 or (step + 1) % self._test_freq == 0:
                test_timer = timer_tools.Timer()
                test_result = evaluator.run_test(self._n_test_episodes,
                        self._test_policy_fn)
                time_cost = test_timer.time_cost()
                test_results.append(
                        [step+1] + list(test_result) + [time_cost])
                self._print_test_info(test_results)
        saver_path = os.path.join(log_dir, 'agent.ckpt')
        self.save_ckpt(saver_path)
        test_results = np.array(test_results)
        np.savetxt(results_file, test_results, fmt='%.4g', delimiter=',')

    def _print_test_info(self, results):
        if len(results) > 0:
            res = results[-1]
            logging.info(
                    'Tested {} episodes at step {}, '
                    'reward mean {:.4g}, std {:.4g}, time cost {:.4g}s.'
                     .format(self._n_test_episodes, res[0], 
                            res[1], res[2], res[3])
            )

    def save_ckpt(self, filepath):
        raise NotImplementedError


class AgentConfig(flag_tools.ConfigBase):

    def _set_default_flags(self):
        flags = self._flags
        flags.device = None
        flags.env_id = None
        # agent
        flags.batch_size = 128
        flags.discount = 0.99
        flags.update_freq = 1
        flags.update_rate = 0.001
        flags.opt_args = None
        # actor args
        flags.actor_cfg = None
        flags.replay_buffer_init = 10000
        flags.replay_buffer_size = int(1e6)
        flags.replay_update_freq = 1
        flags.replay_update_num = 1
        # train
        flags.log_dir = '/tmp/rl_dqn/log'
        flags.total_train_steps = 50000
        flags.print_freq = 1000
        flags.save_freq = 10000
        flags.test_freq = 5000
        flags.n_test_episodes = 50

    def _build(self):
        self._build_env()
        self._build_model()
        self._build_optimizer()
        self._build_args()


    def _obs_prepro(self, obs):
        return obs

    def _env_factory(self):
        raise NotImplementedError

    def _model_factory(self):
        raise NotImplementedError

    def _optimizer_factory(self, parameters):
        raise NotImplementedError

    def _build_env(self):
        dummy_env = self._env_factory()
        dummy_time_step = dummy_env.reset()
        self._action_spec = dummy_env.action_spec
        self._obs_shape = list(self._obs_prepro(
            dummy_time_step.observation).shape)

    def _build_model(self):
        self._model_cfg = flag_tools.Flags(
                model_factory=self._model_factory)

    def _build_optimizer(self):
        self._optimizer_cfg = flag_tools.Flags(
                optimizer_factory=self._optimizer_factory)

    def _build_args(self):
        args = flag_tools.Flags()
        flags = self._flags
        args.device = flags.device
        # env args
        args.action_spec = self._action_spec
        args.obs_shape = self._obs_shape
        args.obs_prepro = self._obs_prepro
        args.env_factory = self._env_factory
        # model args
        args.model_cfg = self._model_cfg
        args.optimizer_cfg = self._optimizer_cfg
        args.batch_size = flags.batch_size
        args.discount = flags.discount
        args.update_freq = flags.update_freq
        args.update_rate = flags.update_rate
        # actor args
        args.actor_cfg = flags.actor_cfg
        args.replay_buffer_init = flags.replay_buffer_init
        args.replay_buffer_size = flags.replay_buffer_size
        args.replay_update_freq = flags.replay_update_freq
        args.replay_update_num = flags.replay_update_num
        # training args
        args.log_dir = flags.log_dir
        args.total_train_steps = flags.total_train_steps
        args.print_freq = flags.print_freq
        args.save_freq = flags.save_freq
        args.test_freq = flags.test_freq
        args.n_test_episodes = flags.n_test_episodes
        self._args = args

    @property
    def args(self):
        return vars(self._args)

    @property
    def args_as_flags(self):
        return self._args








