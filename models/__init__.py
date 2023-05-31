from .vgg_variant import vgg16_variant, vgg16
from .selective_net import SelectiveNet


def get_model(config):
    if config.arch == "selectivenet":
        features = vgg16_variant(32, config.dropout)
        return SelectiveNet(features, config.hidden_state, config.num_classes)
    if config.arch == "vgg16":
        return vgg16(32, config.dropout)

    raise ValueError("Unknown model: {}".format(config.arch))
