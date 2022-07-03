import numpy as np
import collections


# H: horizon, number of transitions.
# h: 1,...,H.
# r: episodic return.
EpisodicStep = collections.namedtuple('EpisodicStep', 'step, h, H, r')


def discounted_sampling(ranges, discount):
    """Draw samples from the discounted distribution over 0, ...., n - 1, 
    where n is a range. The input ranges is a batch of such n`s.

    The discounted distribution is defined as
    p(y = i) = (1 - discount) * discount^i / (1 - discount^n).

    This function implement inverse sampling. We first draw
    seeds from uniform[0, 1) then pass them through the inverse cdf
    floor[ log(1 - (1 - discount^n) * seeds) / log(discount) ]
    to get the samples.
    """
    assert np.min(ranges) >= 1
    assert discount >= 0 and discount <= 1
    seeds = np.random.uniform(size=ranges.shape)
    if discount == 0:
        samples = np.zeros_like(seeds, dtype=np.int64)
    elif discount == 1:
        samples = np.floor(seeds * ranges).astype(np.int64)
    else:
        samples = (np.log(1 - (1 - np.power(discount, ranges)) * seeds) 
                / np.log(discount))
        samples = np.floor(samples).astype(np.int64)
    return samples


def uniform_sampling(ranges):
    return discounted_sampling(ranges, discount=1.0)


class EpisodicReplayBuffer:
    """Only store full episodes.
    
    Sampling returns EpisodicStep objects.
    """

    def __init__(self, max_size):
        self._max_size = max_size
        self._current_size = 0
        self._next_idx = 0
        self._episode_buffer = []
        self._r = 0.0
        self._episodes = []

    @property
    def current_size(self):
        return self._current_size

    @property
    def max_size(self):
        return self._max_size

    def add_steps(self, steps):
        """
        steps: a list of Step(time_step, action, context).
        """
        for step in steps:
            self._episode_buffer.append(step)
            self._r += step.time_step.reward
            # Push each step into the episode buffer until an end-of-episode
            # step is found. 
            # self._r is used to track the cumulative return in each episode.
            if step.time_step.is_last:
                # construct a formal episode
                episode = []
                H = len(self._episode_buffer)
                for h in range(H):
                    epi_step = EpisodicStep(self._episode_buffer[h], 
                            h + 1, H, self._r)
                    episode.append(epi_step)
                # save as data
                if self._next_idx == self._current_size:
                    if self._current_size < self._max_size:
                        self._episodes.append(episode)
                        self._current_size += 1
                        self._next_idx += 1
                    else:
                        self._episodes[0] = episode
                        self._next_idx = 1
                else:
                    self._episodes[self._next_idx] = episode
                    self._next_idx += 1
                # refresh episode buffer
                self._episode_buffer = []
                self._r = 0.0

    def sample_steps(self, batch_size):
        episode_indices = self._sample_episodes(batch_size)
        step_ranges = self._gather_episode_lengths(episode_indices)
        step_indices = uniform_sampling(step_ranges)
        s = []
        for epi_idx, step_idx in zip(episode_indices, step_indices):
            s.append(self._episodes[epi_idx][step_idx])
        return s

    def sample_transitions(self, batch_size):
        episode_indices = self._sample_episodes(batch_size)
        step_ranges = self._gather_episode_lengths(episode_indices)
        step_indices = uniform_sampling(step_ranges - 1)
        s1 = []
        s2 = []
        for epi_idx, step_idx in zip(episode_indices, step_indices):
            s1.append(self._episodes[epi_idx][step_idx])
            s2.append(self._episodes[epi_idx][step_idx + 1])
        return s1, s2

    def sample_pairs(self, batch_size, discount=0.0):
        episode_indices = self._sample_episodes(batch_size)
        step_ranges = self._gather_episode_lengths(episode_indices)
        step1_indices = uniform_sampling(step_ranges - 1)
        intervals = discounted_sampling(
            step_ranges - step1_indices - 1, discount=discount) + 1
        step2_indices = step1_indices + intervals
        s1 = []
        s2 = []
        for epi_idx, step1_idx, step2_idx in zip(
                episode_indices, step1_indices, step2_indices):
            s1.append(self._episodes[epi_idx][step1_idx])
            s2.append(self._episodes[epi_idx][step2_idx])
        return s1, s2

    def _sample_episodes(self, batch_size):
        return np.random.randint(self._current_size, size=batch_size)

    def _gather_episode_lengths(self, episode_indices):
        lengths = []
        for index in episode_indices:
            lengths.append(len(self._episodes[index]))
        return np.array(lengths, dtype=np.int64)

