"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
"""


import types
import importlib

class LazyLoader(types.ModuleType):
  """Lazily import a module, mainly to avoid pulling in large dependencies.

  `paddle`, and `ffmpeg` are examples of modules that are large and not always
  needed, and this allows them to only be loaded when they are used.
  """

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name, warning=None):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    self._warning = warning

    # These members allows doctest correctly process this module member without
    # triggering self._load(). self._load() mutates parant_module_globals and
    # triggers a dict mutated during iteration error from doctest.py.
    # - for from_module()
    self.__module__ = name.rsplit(".", 1)[0]
    # - for is_routine()
    self.__wrapped__ = None

    super(LazyLoader, self).__init__(name)

  def _load(self):
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module

    # Emit a warning if one was specified
    if self._warning:
      # logging.warning(self._warning)
      # Make sure to only warn once.
      self._warning = None

    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)

    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __repr__(self):
    # Carefully to not trigger _load, since repr may be called in very
    # sensitive places.
    return f"<LazyLoader {self.__name__} as {self._local_name}>"

  def __dir__(self):
    module = self._load()
    return dir(module)


# import paddle.nn as nn
# nn = LazyLoader("nn", globals(), "paddle.nn")

# class M(nn.Layer):
#     def __init__(self) -> None:
#       super().__init__()
