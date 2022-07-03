import time


class Timer:

    def __init__(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def reset(self):
        self._start_time = time.time()
        self._step_time = time.time()
        self._step = 0

    def set_step(self, step):
        self._step = step
        self._step_time = time.time()

    def time_cost(self):
        return time.time() - self._start_time

    def steps_per_sec(self, step):
        sps = (step - self._step) / (time.time() - self._step_time)
        self._step = step
        self._step_time = time.time()
        return sps