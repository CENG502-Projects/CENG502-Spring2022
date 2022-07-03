import gym
from . import env_base


class GymTask(env_base.Task):

    def __init__(self, env_id):
        self._env_id = env_id
        self._env = gym.make(self._env_id)
        self._last_step = [None, 0.0, False, {}]  # obs, r, done, info
        self._has_timelimit = isinstance(self._env, gym.wrappers.TimeLimit)

    @property
    def env(self):
        return self._env

    def begin_episode(self):
        r = self._last_step[1]
        self._last_step = [self._env.reset(), r, False, 0]

    def step(self, action):
        #if self._last_step[2]:
        #    self._last_step = self._env.reset()
        #else:
        self._last_step = list(self._env.step(action))

    def get_observation(self):
        return self._last_step[0]

    def get_reward(self):
        """check reward is not None?"""
        return self._last_step[1]

    def get_info(self):
        return self._last_step[3]

    def is_end_episode(self):
        if self._has_timelimit:
            return (not self.past_timelimit()) and self._last_step[2]
        else:
            return self._last_step[2]

    def past_timelimit(self):
        if self._has_timelimit:
            # return self._env._past_limit()  
            # the above line doesn't work for newer version of gym
            # because of a stupid change
            key = 'TimeLimit.truncated'
            info = self._last_step[3]
            return (key in info) and info[key]
        else:
            return False

    @property
    def action_spec(self):
        action_space = self._env.action_space
        if isinstance(action_space, gym.spaces.Box):
            action_spec = env_base.ContinuousActionSpec(
                low=action_space.low,
                high=action_space.high
            )
        elif isinstance(action_space, gym.spaces.Discrete):
            action_spec = env_base.DiscreteActionSpec(action_space.n)
        else:
            raise ValueError('Unknown action space type.')
        return action_spec


class Environment(env_base.Environment):

    def __init__(self, env_id):
        task = GymTask(env_id)
        super().__init__(task)

