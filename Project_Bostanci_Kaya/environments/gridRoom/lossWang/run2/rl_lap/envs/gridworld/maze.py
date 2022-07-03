import numpy as np


DEFAULT_MAZE = '''
+---------+-------+
|         |       |
| +-------+ +---+ +
|           |   | |
| +-----+---+ + + |
| |     |   | |   |
| | +-+ + + | +-+ |
| | |   | |   |   |
| + | +-+ +---+ +-+
|   |   | |   |   |
| +-+-+ | + + | + |
| |     |   | | | |
| | +-+-+-+-+ | +-+
| | |     |   |   |
| +-+ +-+ | +-+---+
| |   | | |       |
| + + + | +---+-+ |
|   |   |     |   |
+-------+-+---+---+
'''


HARD_MAZE = '''
+----+----+----+----+
|    +    |    +    |
|         +         |
|    +         +    |
|    |    +    |    |
+----+----+----+-+ ++
|    +    |    |    |
|         +    |    |
|    +         |    |
|    |    +    |    |
+-+ ++----+-+ +++ +-+
|    |    |    |    |
|    |    +    |    |
|    |         |    |
|    |    +    |    |
++ +-+----+----+-+ ++
|    |    +    |    |
|    +         +    |
|         +         |
|    +    |    +    |
+----+----+----+----+
'''


class MazeFactoryBase:
    def __init__(self, maze_str=DEFAULT_MAZE):
        self._maze = self._parse_maze(maze_str)

    def _parse_maze(self, maze_source):
        width = 0
        height = 0
        maze_matrix = []
        for row in maze_source.strip().split('\n'):
            row_vector = row.strip()
            maze_matrix.append(row_vector)
            height += 1
            width = max(width, len(row_vector))
        maze_array = np.zeros([height, width], dtype=str)
        maze_array[:] = ' '
        for i, row in enumerate(maze_matrix):
            for j, val in enumerate(row):
                maze_array[i, j] = val
        return maze_array

    def get_maze(self):
        return self._maze


class SquareRoomFactory(MazeFactoryBase):
    """generate a square room with given size"""
    def __init__(self, size):
        maze_array = np.zeros([size+2, size+2], dtype=str)
        maze_array[:] = ' '
        maze_array[0] = '-'
        maze_array[-1] = '-'
        maze_array[:, 0] = '|'
        maze_array[:, -1] = '|'
        maze_array[0, 0] = '+'
        maze_array[0, -1] = '+'
        maze_array[-1, 0] = '+'
        maze_array[-1, -1] = '+'
        self._maze = maze_array


class FourRoomsFactory(MazeFactoryBase):
    """generate four rooms, each with the given size"""
    def __init__(self, size):
        maze_array = np.zeros([size*2+3, size*2+3], dtype=str)
        maze_array[:] = ' '
        wall_idx = [0, size+1, size*2+2]
        maze_array[wall_idx] = '-'
        maze_array[:, wall_idx] = '|'
        maze_array[wall_idx][:, wall_idx] = '+'
        door_idx = [int((size+1)/2), int((size+1)/2)+1, 
                int((size+1)/2)+size+1, int((size+1)/2)+size+2]
        maze_array[size+1, door_idx] = ' '
        maze_array[door_idx, size+1] = ' '
        self._maze = maze_array


class TwoRoomsFactory(MazeFactoryBase):
    def __init__(self, size):
        maze_array = np.zeros([size+2, size+2], dtype=str)
        maze_array[:] = ' '
        hwall_idx = [0, int((size+1)/2), size+1]
        vwall_idx = [0, size+1]
        maze_array[hwall_idx] = '-'
        maze_array[:, vwall_idx] = '|'
        maze_array[hwall_idx][:, vwall_idx] = '+'
        door_idx = [int((size+1)/2), int((size+1)/2)+1]
        maze_array[hwall_idx[1], door_idx] = ' '
        self._maze = maze_array


class Maze:
    def __init__(self, maze_factory):
        self._maze_factory = maze_factory
        # parse maze ...
        self._maze = None
        self._height = None
        self._width = None
        self._build_maze()
        self._all_empty_grids = np.argwhere(self._maze==' ')
        self._n_states = self._all_empty_grids.shape[0]
        self._pos_indices = {}
        for i, pos in enumerate(self._all_empty_grids):
            self._pos_indices[tuple(pos)] = i

    def _build_maze(self):
        self._maze = self._maze_factory.get_maze()
        self._height = self._maze.shape[0]
        self._width = self._maze.shape[1]

    def rebuild(self):
        self._build_maze()

    def __getitem__(self, key):
        return self._maze[key]

    def __setitem__(self, key, val):
        self._maze[key] = val

    def is_empty(self, pos):
        if (pos[0] >= 0 and pos[0] < self._height 
                and pos[1] >= 0 and pos[1] < self._width):
            return self._maze[tuple(pos)] == ' '
        else:
            return False
    
    @property
    def maze_array(self):
        return self._maze

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def n_states(self):
        return self._n_states

    def pos_index(self, pos):
        return self._pos_indices[tuple(pos)]

    def all_empty_grids(self):
        return np.argwhere(self._maze==' ')

    def random_empty_grids(self, k):
        '''Return k random empty positions.'''
        empty_grids = np.argwhere(self._maze==' ')
        selected = np.random.choice(
                np.arange(empty_grids.shape[0]),
                size=k,
                replace=False
                )
        return empty_grids[selected]

    def first_empty_grid(self):
        empty_grids = np.argwhere(self._maze==' ')
        assert empty_grids.shape[0] > 0
        return empty_grids[0]

    def render(self):
        # 0 for ground, 1 for wall
        return (self._maze!=' ').astype(np.float32)


