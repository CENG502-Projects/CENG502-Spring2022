import numpy as np
import collections

from .. import env_base


ObservationType = collections.namedtuple(
        'ObservationType', 'image, position, index')


AGENT_COLOR = (1., 0., 0.)
WALL_COLOR = (0., 0., 0.)
GROUND_COLOR = (1., 1., 1.)


def one_hot(x, n):
    if len(np.shape(x)) == 0:
        res = np.zeros([n])
        res[x] = 1
        return res
    else:
        b = x.shape[0]
        res = np.zeros([b, n])
        res[np.arange(b), x] = 1
        return res


def is_same_pos(pos1, pos2):
    return pos1[0] == pos2[0] and pos1[1] == pos2[1]


class Maze2DBase(env_base.Task):

    def __init__(
            self, 
            maze, 
            episode_len=50, 
            start_pos='first', 
            use_stay_action=True,
            ):
        self._maze = maze
        self._episode_len = episode_len
        if type(start_pos) == str:
            assert start_pos in ['first', 'random']
        self._start_pos = start_pos
        if use_stay_action:
            self._action_map = np.array(
                    [[-1, 0], 
                    [1, 0], 
                    [0, -1], 
                    [0, 1], 
                    [0, 0]])
        else:
            self._action_map = np.array(
                [[-1, 0], 
                [1, 0], 
                [0, -1], 
                [0, 1]])
        self._action_spec = env_base.DiscreteActionSpec(
                self._action_map.shape[0])
        # to be maintained during each episode
        self._agent_pos = None # a numpy array with shape (2,)
        self._steps_taken = None # number of steps taken so far
        self._should_end_episode = None

    def begin_episode(self):
        self._maze.rebuild()
        if type(self._start_pos) == str:
            if self._start_pos == 'first':
                self._agent_pos = self._maze.first_empty_grid()
            elif self._start_pos == 'random':
                self._agent_pos = self._maze.random_empty_grids(1)[0]
        else:
            self._agent_pos = self._start_pos.copy()
        self._steps_taken = 0
        self._should_end_episode = False

    def step(self, action):
        assert not self._should_end_episode
        self._steps_taken += 1
        new_agent_pos = self._agent_pos + self._action_map[action]
        if self._maze.is_empty(new_agent_pos):
            self._agent_pos = new_agent_pos

    def get_observation(self):
        return self.pos_to_obs(self._agent_pos)

    def pos_to_obs(self, pos):
        # render a H*W*3 colored map
        binary_maze = self._maze.render()
        walls = np.outer(binary_maze.flatten(), np.array(WALL_COLOR))
        ground = np.outer(1-binary_maze.flatten(), np.array(GROUND_COLOR))
        colored_maze = (walls + ground).reshape(
            [self._maze.height, self._maze.width, 3])
        colored_maze[tuple(pos)] = np.array(AGENT_COLOR)
        pos_idx = one_hot(self._maze.pos_index(pos), self._maze.n_states)
        return ObservationType(
            image=colored_maze,
            position=self.normalize_pos(pos),
            index=pos_idx)

    def normalize_pos(self, pos):
        maze_shape = self._maze.maze_array.shape
        x = pos[0] / maze_shape[0] - 0.5
        y = pos[1] / maze_shape[1] - 0.5
        return np.array([x, y])

    def get_reward(self):
        return 0.0

    def get_info(self):
        return self._maze.pos_index(self._agent_pos)

    def is_end_episode(self):
        return self._should_end_episode

    def past_timelimit(self):
        return self._steps_taken >= self._episode_len

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def maze(self):
        return self._maze

    @property
    def n_states(self):
        return self._maze.n_states

    def render_maze(self):
        return self._maze.render()

