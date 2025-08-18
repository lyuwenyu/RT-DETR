""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import inspect
import importlib
import functools
import inspect
from collections import defaultdict
from typing import Any, Dict, Optional, List


GLOBAL_CONFIG = defaultdict(dict)


def register(dct :Any=GLOBAL_CONFIG, name=None, force=False):
    """
        dct:
            if dct is Dict, register foo into dct as key-value pair
            if dct is Clas, register as modules attibute
        force 
            whether force register.
    """
    def decorator(foo):
        register_name = foo.__name__ if name is None else name
        if not force:
            if inspect.isclass(dct):
                assert not hasattr(dct, foo.__name__), \
                    f'module {dct.__name__} has {foo.__name__}'
            else:
                assert foo.__name__ not in dct, \
                f'{foo.__name__} has been already registered'

        if inspect.isfunction(foo):
            @functools.wraps(foo)
            def wrap_func(*args, **kwargs):
                return foo(*args, **kwargs)
            if isinstance(dct, dict):
                dct[foo.__name__] = wrap_func
            elif inspect.isclass(dct):
                setattr(dct, foo.__name__, wrap_func)
            else:
                raise AttributeError('')
            return wrap_func

        elif inspect.isclass(foo):
            dct[register_name] = extract_schema(foo) 

        else:
            raise ValueError(f'Do not support {type(foo)} register')

        return foo

    return decorator



def extract_schema(module: type):
    """
    Args:
        module (type),
    Return:
        Dict, 
    """
    argspec = inspect.getfullargspec(module.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    num_defualts = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defualts

    schame = dict()
    schame['_name'] = module.__name__
    schame['_pymodule'] = importlib.import_module(module.__module__)
    schame['_inject'] = getattr(module, '__inject__', [])
    schame['_share'] = getattr(module, '__share__', [])
    schame['_kwargs'] = {}
    for i, name in enumerate(arg_names):
        if name in schame['_share']:
            assert i >= num_requires, 'share config must have default value.'
            value = argspec.defaults[i - num_requires]
        
        elif i >= num_requires:
            value = argspec.defaults[i - num_requires]

        else:
            value = None 

        schame[name] = value
        schame['_kwargs'][name] = value 
        
    return schame


def create(type_or_name, global_cfg=GLOBAL_CONFIG, **kwargs):
    """
    """
    assert type(type_or_name) in (type, str), 'create should be modules or name.'

    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    if name in global_cfg:
        if hasattr(global_cfg[name], '__dict__'):
            return global_cfg[name]
    else:
        raise ValueError('The module {} is not registered'.format(name))

    cfg = global_cfg[name]

    if isinstance(cfg, dict) and 'type' in cfg:
        _cfg: dict = global_cfg[cfg['type']]
        # clean args
        _keys = [k for k in _cfg.keys() if not k.startswith('_')]
        for _arg in _keys:
            del _cfg[_arg]
        _cfg.update(_cfg['_kwargs']) # restore default args
        _cfg.update(cfg) # load config args 
        _cfg.update(kwargs) # TODO recive extra kwargs
        name = _cfg.pop('type') # pop extra key `type` (from cfg)
        
        return create(name, global_cfg)
    
    module = getattr(cfg['_pymodule'], name)    
    module_kwargs = {}
    module_kwargs.update(cfg)
    
    # shared var
    for k in cfg['_share']:
        if k in global_cfg:
            module_kwargs[k] = global_cfg[k]
        else:
            module_kwargs[k] = cfg[k]

    # inject
    for k in cfg['_inject']:
        _k = cfg[k]

        if _k is None:
            continue

        if isinstance(_k, str):            
            if _k not in global_cfg:
                raise ValueError(f'Missing inject config of {_k}.')

            _cfg = global_cfg[_k]
            
            if isinstance(_cfg, dict):
                module_kwargs[k] = create(_cfg['_name'], global_cfg)
            else:
                module_kwargs[k] = _cfg 

        elif isinstance(_k, dict):
            if 'type' not in _k.keys():
                raise ValueError(f'Missing inject for `type` style.')

            _type = str(_k['type'])
            if _type not in global_cfg:
                raise ValueError(f'Missing {_type} in inspect stage.')

            # TODO 
            _cfg: dict = global_cfg[_type]
            # clean args
            _keys = [k for k in _cfg.keys() if not k.startswith('_')]
            for _arg in _keys:
                del _cfg[_arg]
            _cfg.update(_cfg['_kwargs']) # restore default values
            _cfg.update(_k) # load config args
            name = _cfg.pop('type') # pop extra key (`type` from _k)
            module_kwargs[k] = create(name, global_cfg)

        else:
            raise ValueError(f'Inject does not support {_k}')
    
    # TODO hard code
    module_kwargs = {k: v for k, v in module_kwargs.items() if not k.startswith('_')}

    # TODO for **kwargs
    # extra_args = set(module_kwargs.keys()) - set(arg_names)
    # if len(extra_args) > 0:
    #     raise RuntimeError(f'Error: unknown args {extra_args} for {module}')

    return module(**module_kwargs)