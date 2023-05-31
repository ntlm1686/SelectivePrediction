from .selective import SelectiveModule
from .default import DefaultModule


def create(config, len_train=None):
    if config.mode == "selective":
        return SelectiveModule(config, len_train)
    if config.mode == "default":
        return DefaultModule(config, len_train)
    else:
        raise ValueError("Unknown module: {}".format(config.module))