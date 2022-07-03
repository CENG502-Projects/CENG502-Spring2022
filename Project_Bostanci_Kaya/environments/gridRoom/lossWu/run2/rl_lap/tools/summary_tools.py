import numpy as np


def get_summary_str(step=None, info=None, prefix=''):
    summary_str = prefix
    if step is not None:
        summary_str += 'Step {}; '.format(step)
    for key, val in info.items():
        if isinstance(val, (int, np.int32, np.int64)):
            summary_str += '{} {}; '.format(key, val)
        elif isinstance(val, (float, np.float32, np.float64)):
            summary_str += '{} {:.4g}; '.format(key, val)
    return summary_str


def write_summary(writer, info, step):
    """For pytorch. Write summary to tensorboard."""
    for key, val in info.items():
        if isinstance(val, 
                (int, float, np.int32, np.int64, np.float32, np.float64)):
            writer.add_scalar(key, val, step)