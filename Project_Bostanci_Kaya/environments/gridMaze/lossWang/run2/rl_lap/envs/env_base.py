import collections
import numpy as np

# is final: terminal state
# is last: last step in an episode, may not be terminal state
TimeStep = collections.namedtuple('TimeStep', 
        'observation, reward, is_final, is_last, info')

class ContinuousActionSpec:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)

    def sample_batch(self, size):
        size = [size, self.low.shape[0]]
        return np.random.uniform(self.low, self.high, size=size)


class DiscreteActionSpec:

    def __init__(self, n):
        self.n = n
    
    def sample(self):
        return np.random.randint(self.n)

    def sample_batch(self, size):
        return np.random.randint(self.n, size=size)


class Task:

    def begin_episode(self):
        """
        Begin a new episode, no need to return, but the first TimeStep
        should be available.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Take a step and update the TimeStep.
        """
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_info(self):
        return None

    def is_end_episode(self):
        """
        True or False: the last TimeStep is the end of an episode.
        """
        raise NotImplementedError

    def past_timelimit(self):
        raise NotImplementedError

    @property
    def action_spec(self):
        raise NotImplementedError


class Environment:

    def __init__(self, task):
        self._task = task
        self._should_restart = True
        self._last_step = self.reset()
        # this will lead to another reset at the first step,
        # so that the first action acts as a reset
        self._should_restart = True  

    @property
    def task(self):
        return self._task

    def reset(self):
        self._task.begin_episode()
        obs = self._task.get_observation()
        reward = self._task.get_reward()  # should be 0?
        info = self._task.get_info()
        self._last_step = TimeStep(obs, reward, False, False, info)
        self._should_restart = False
        return self._last_step

    def step(self, action):
        if self._should_restart:
            return self.reset()
        self._task.step(action)
        obs = self._task.get_observation()
        reward = self._task.get_reward()
        info = self._task.get_info()
        is_final = self._task.is_end_episode()
        past_timelimit = self._task.past_timelimit() 
        if is_final:
            is_final = True
            is_last = True
            self._should_restart = True
        elif past_timelimit:
            is_final = False
            is_last = True
            self._should_restart = True
        else:
            is_final = False
            is_last = False
        self._last_step = TimeStep(obs, reward, is_final, is_last, info)
        return self._last_step

    @property
    def is_end_episode(self):
        return self._should_restart

    @property
    def action_spec(self):
        return self._task.action_spec
        

