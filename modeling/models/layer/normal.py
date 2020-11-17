import functools
import torch.nn as nn


class LayerManager:
    @staticmethod
    def set_norm_layer(norm_type="instance"):
        """
        Args:
            norm_type (str) -- the name of the normalization layer: batch | instance | none
        return:
        """
        if norm_type == "batch":
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == "none":
            norm_layer = None
        else:
            raise NotImplementedError("normalization layer '{}' is not found".format(norm_type))
        return norm_layer



