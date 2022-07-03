import numpy as np


class Evaluator:

    def __init__(self, env_factory, max_ep_len=10000):
        self._env_factory = env_factory
        self._max_ep_len = max_ep_len

    def run_test(self, n_episodes, policy_fn):
        n = n_episodes
        env = self._env_factory()
        r = np.zeros(n)
        for i in range(n):
            r_tmp = 0
            time_step = env.reset()
            step = 0
            context = None
            while (not env.is_end_episode and step <= self._max_ep_len):
                state = (time_step, context)
                a, context = policy_fn(state)
                time_step = env.step(a)
                r_tmp += time_step.reward
            r[i] = r_tmp
        r_mean = np.mean(r)
        r_std = np.std(r)
        return r_mean, r_std


