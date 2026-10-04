import logging

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader


class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = {}

    def get(self, key):
        return self._module_dict.get(key)

    def register_module(self, cls=None, name=None, force=False, module=None):
        def _register(obj):
            key = name or obj.__name__
            if key in self._module_dict and not force:
                raise KeyError(f"{key} is already registered in {self.name}")
            self._module_dict[key] = obj
            return obj

        if cls is None and module is None:
            return _register
        return _register(cls if cls is not None else module)


def build_from_cfg(cfg, registry, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")
    args = cfg.copy()
    if default_args is not None:
        for key, value in default_args.items():
            args.setdefault(key, value)
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    else:
        obj_cls = obj_type
    return obj_cls(**args)


def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(log_level)
    return logger


def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        return
    else:
        raise TypeError(f"logger should be a Logger, None, or 'silent', got {type(logger)}")


def collect_env():
    return {}


def get_git_hash():
    return "unknown"


__all__ = [
    "Registry",
    "build_from_cfg",
    "get_logger",
    "print_log",
    "collect_env",
    "get_git_hash",
    "_BatchNorm",
    "DataLoader",
]
