import inspect
import functools

def store_args(fn):
    """
    Decorator for saving arguments in __init__ as class attributes.
    """
    fn_args = inspect.signature(fn).parameters
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        self_ = args[0]
        arg_keys = list(fn_args.keys())[1:]
        for val, key in zip(args[1:], arg_keys):
            setattr(self_, '_'+key, val)
        n_pos_args = len(args) - 1
        for key in arg_keys[n_pos_args:]:
            if key in kwargs:
                setattr(self_, '_'+key, kwargs[key])
            else:
                setattr(self_, '_'+key, fn_args[key].default)
        return fn(*args, **kwargs)
    return wrapper


def store_attrs(fn):
    """Same as store_args, but without prefix '_'."""
    fn_args = inspect.signature(fn).parameters
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        self_ = args[0]
        arg_keys = list(fn_args.keys())[1:]
        for val, key in zip(args[1:], arg_keys):
            setattr(self_, key, val)
        n_pos_args = len(args) - 1
        for key in arg_keys[n_pos_args:]:
            if key in kwargs:
                setattr(self_, key, kwargs[key])
            else:
                setattr(self_, key, fn_args[key].default)
        return fn(*args, **kwargs)
    return wrapper

