""""by lyuwenyu
"""

import os
import yaml 
import inspect
import importlib

__all__ = ['GLOBAL_CONFIG', 'register', 'create', 'load_config', 'merge_config', 'merge_dict']


GLOBAL_CONFIG = dict()
INCLUDE_KEY = '__include__'


def register(cls: type):
    '''
    Args:
        cls (type): Module class to be registered.
    '''
    if cls.__name__ in GLOBAL_CONFIG:
        raise ValueError('{} already registered'.format(cls.__name__))

    if inspect.isfunction(cls):
        GLOBAL_CONFIG[cls.__name__] = cls
    
    elif inspect.isclass(cls):
        GLOBAL_CONFIG[cls.__name__] = extract_schema(cls)

    else:
        raise ValueError(f'register {cls}')

    return cls 


def extract_schema(cls: type):
    '''
    Args:
        cls (type),
    Return:
        Dict, 
    '''
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    num_defualts = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defualts

    schame = dict()
    schame['_name'] = cls.__name__
    schame['_pymodule'] = importlib.import_module(cls.__module__)
    schame['_inject'] = getattr(cls, '__inject__', [])
    schame['_share'] = getattr(cls, '__share__', [])

    for i, name in enumerate(arg_names):
        if name in schame['_share']:
            assert i >= num_requires, 'share config must have default value.'
            value = argspec.defaults[i - num_requires]
        
        elif i >= num_requires:
            value = argspec.defaults[i - num_requires]

        else:
            value = None 

        schame[name] = value
        
    return schame



def create(type_or_name, **kwargs):
    '''
    '''
    assert type(type_or_name) in (type, str), 'create should be class or name.'

    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    if name in GLOBAL_CONFIG:
        if hasattr(GLOBAL_CONFIG[name], '__dict__'):
            return GLOBAL_CONFIG[name]
    else:
        raise ValueError('The module {} is not registered'.format(name))

    cfg = GLOBAL_CONFIG[name]

    if isinstance(cfg, dict) and 'type' in cfg:
        _cfg: dict = GLOBAL_CONFIG[cfg['type']]
        _cfg.update(cfg) # update global cls default args 
        _cfg.update(kwargs) # TODO
        name = _cfg.pop('type')
        
        return create(name)


    cls = getattr(cfg['_pymodule'], name)
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    
    cls_kwargs = {}
    cls_kwargs.update(cfg)
    
    # shared var
    for k in cfg['_share']:
        if k in GLOBAL_CONFIG:
            cls_kwargs[k] = GLOBAL_CONFIG[k]
        else:
            cls_kwargs[k] = cfg[k]

    # inject
    for k in cfg['_inject']:
        _k = cfg[k]

        if _k is None:
            continue

        if isinstance(_k, str):            
            if _k not in GLOBAL_CONFIG:
                raise ValueError(f'Missing inject config of {_k}.')

            _cfg = GLOBAL_CONFIG[_k]
            
            if isinstance(_cfg, dict):
                cls_kwargs[k] = create(_cfg['_name'])
            else:
                cls_kwargs[k] = _cfg 

        elif isinstance(_k, dict):
            if 'type' not in _k.keys():
                raise ValueError(f'Missing inject for `type` style.')

            _type = str(_k['type'])
            if _type not in GLOBAL_CONFIG:
                raise ValueError(f'Missing {_type} in inspect stage.')

            # TODO modified inspace, maybe get wrong result for using `> 1`
            _cfg: dict = GLOBAL_CONFIG[_type]
            # _cfg_copy = copy.deepcopy(_cfg)
            _cfg.update(_k) # update 
            cls_kwargs[k] = create(_type)
            # _cfg.update(_cfg_copy) # resume

        else:
            raise ValueError(f'Inject does not support {_k}')


    cls_kwargs = {n: cls_kwargs[n] for n in arg_names}

    return cls(**cls_kwargs)



def load_config(file_path, cfg=dict()):
    '''load config
    '''
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        if file_cfg is None:
            return {}

    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)

            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            with open(base_yaml) as f:
                base_cfg = load_config(base_yaml, cfg)
                merge_config(base_cfg, cfg)

    return merge_config(file_cfg, cfg)



def merge_dict(dct, another_dct):
    '''merge another_dct into dct
    '''
    for k in another_dct:
        if (k in dct and isinstance(dct[k], dict) and isinstance(another_dct[k], dict)):
            merge_dict(dct[k], another_dct[k])
        else:
            dct[k] = another_dct[k]

    return dct



def merge_config(config, another_cfg=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.

    Returns: global config
    """
    global GLOBAL_CONFIG
    dct = GLOBAL_CONFIG if another_cfg is None else another_cfg
    
    return merge_dict(dct, config)



