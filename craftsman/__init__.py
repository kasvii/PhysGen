import importlib

__modules__ = {}


def clear_registry():
    """Clear the module registry to avoid conflicts during re-imports."""
    global __modules__
    __modules__.clear()


def register(name):
    def decorator(cls):
        if name in __modules__:
            # If the same class is registered again, just return it
            if __modules__[name] is cls:
                return cls
            # Check if it's the same class from a different module path
            existing_cls = __modules__[name]
            if (hasattr(existing_cls, '__name__') and hasattr(cls, '__name__') and
                existing_cls.__name__ == cls.__name__ and
                existing_cls.__qualname__ == cls.__qualname__):
                return cls
            raise ValueError(
                f"Module {name} already exists! Names of extensions conflict!"
            )
        else:
            __modules__[name] = cls
        return cls

    return decorator


def find(name):
    if name in __modules__:
        return __modules__[name]
    else:
        try:
            module_string = ".".join(name.split(".")[:-1])
            cls_name = name.split(".")[-1]
            module = importlib.import_module(module_string, package=None)
            return getattr(module, cls_name)
        except Exception as e:
            raise ValueError(f"Module {name} not found!")


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("pytorch_lightning")

from pytorch_lightning.utilities.rank_zero import (
    rank_zero_debug,
    rank_zero_info,
    rank_zero_only,
)

debug = rank_zero_debug
info = rank_zero_info


@rank_zero_only
def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


from . import data, models, systems
