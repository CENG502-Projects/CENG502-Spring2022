import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize

import pickle
from functools import partial
import numpy as np
import PIL
import random
from pathlib import Path
from distutils.dir_util import copy_tree
import shutil


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def load_checkpoint(fpath):
    if fpath is None:
        raise ValueError('File path is None')
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = "cuda:0" if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(str(fpath), map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(module, is_trainable):
    apply_leaf(module, lambda m: set_trainable_attr(m, is_trainable))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def code_backup(to_path):
    lib_dir = Path(__file__).resolve().parent.parent
    conf_dir = lib_dir.parent / "configs"
    train_file = lib_dir.parent / "train.py"

    copy_tree(str(lib_dir), str(to_path / "lib"))
    copy_tree(str(conf_dir), str(to_path / "configs"))
    shutil.copyfile(str(train_file), str(to_path / "train.py"))