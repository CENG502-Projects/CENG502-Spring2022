import numpy as np
import torch


def to_tensor(x, device):
    """return a torch.Tensor, assume x is an np array."""
    if x.dtype in [np.float32, np.float64]:
        return torch.tensor(x, dtype=torch.float32, device=device)
    elif x.dtype in [np.int32, np.int64, np.uint8]:
        return torch.tensor(x, dtype=torch.int64, device=device)
    else:
        raise ValueError('Unknown dtype {}.'.format(str(x.dtype)))