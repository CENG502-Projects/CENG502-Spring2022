import os
import yaml
import ast


def type_bool(x):
    return x.lower() != 'false'


class Flags:

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def update_flags(flags, updates):
    '''
    Flags maybe hierachical.
    Updates come in argparse flags format:
    Key "x.y" for updating flags.x.y.
    '''
    updates_dict = vars(updates)
    for key, val in updates_dict.items():
        if key[0] == '_':
            continue
        subkeys = key.split('.')
        current_flags = flags
        has_flag = True
        for subkey in subkeys[:-1]:
            if hasattr(current_flags, subkey):
                current_flags = getattr(current_flags, subkey)
            else:
                has_flag = False
                break
        if has_flag and hasattr(current_flags, subkeys[-1]):
            setattr(current_flags, subkeys[-1], val)


def flags_to_dict(flags):
    dict_ = vars(flags).copy()
    for key, val in dict_.items():
        if hasattr(val, '__dict__'):
            dict_[key] = flags_to_dict(val)
    return dict_


def dict_to_flags(dict_):
    flags = Flags()
    flags.__dict__.update(dict_)
    for key, val in vars(flags).items():
        if isinstance(val, dict):
            setattr(flags, key, dict_to_flags(val))
    return flags


def save_flags(flags, log_dir, filename='flags.yaml'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'w') as f:
        yaml.dump(flags_to_dict(flags), f)


def load_flags(log_dir='', filename='flags.yaml'):
    filepath = os.path.join(log_dir, filename)
    with open(filepath, 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
    return dict_to_flags(d)


def auto_type(val):
    """Cast val(type=str) into detected type."""
    return ast.literal_eval(val)


def parse_args(flags, keyword='args'):
    """
    Assume flags.args is a list of additional flags 
        e.g. not explicitly defined by argparse.
    """
    new_flags = Flags()
    for key, val in vars(flags).items():
        if key != keyword:
            setattr(new_flags, key, val)
        else:
            try:
                for arg in val:
                    arg_key, arg_val = arg.split('=')
                    arg_val = auto_type(arg_val)
                    setattr(new_flags, arg_key, arg_val)
            except ValueError as err_msg:
                raise ValueError('Invalid format of args {}. '
                        'Acceptable format: --args="key=val".\n'
                        'Error message: {}'
                        .format(val, err_msg))
    return new_flags


class ConfigBase:

    def __init__(self, flags):
        self._flags = Flags()
        self._set_default_flags()
        parsed_flags = parse_args(flags)
        update_flags(self._flags, parsed_flags)
        self._build()

    def _set_default_flags(self):
        """Set flags that can be updated later."""
        pass

    def _build(self):
        pass

    @property
    def flags(self):
        return self._flags

    @property
    def flags_dict(self):
        return flags_to_dict(self._flags)

    def save_flags(self, log_dir, filename='flags.yaml'):
        save_flags(self._flags, log_dir, filename)

