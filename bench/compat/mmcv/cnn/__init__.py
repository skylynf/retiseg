"""The conv and norm constructors used by the author's HRNet and FCN head."""

import torch.nn as nn

from mmcv.utils import Registry

UPSAMPLE_LAYERS = Registry("upsample layer")


def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        layer_type = "Conv2d"
    else:
        cfg = cfg.copy()
        layer_type = cfg.pop("type", "Conv2d")
    if layer_type != "Conv2d":
        raise KeyError(layer_type)
    return nn.Conv2d(*args, **kwargs)


def build_norm_layer(cfg, num_features, postfix=""):
    if not isinstance(cfg, dict):
        raise TypeError(f"norm cfg must be a dict, got {type(cfg)}")
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    if layer_type not in ("BN", "BN2d", "SyncBN"):
        raise KeyError(layer_type)
    requires_grad = cfg.pop("requires_grad", True)
    cfg.setdefault("eps", 1e-5)
    cfg.setdefault("momentum", 0.1)
    # The checkpoint was trained with mmcv SyncBN. At evaluation a single
    # process uses the stored running mean and variance, which BatchNorm2d
    # applies with the same affine formula.
    layer = nn.BatchNorm2d(num_features, **cfg)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return "bn" + str(postfix), layer


def build_plugin_layer(cfg, *args, **kwargs):
    raise NotImplementedError("plugin layers are not part of the M2MRF-C checkpoint")


def build_upsample_layer(cfg, *args, **kwargs):
    raise NotImplementedError("upsample layers are not part of the M2MRF-C checkpoint")


def build_activation_layer(cfg):
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    if layer_type == "ReLU":
        return nn.ReLU(**cfg)
    if layer_type == "LeakyReLU":
        return nn.LeakyReLU(**cfg)
    if layer_type == "Sigmoid":
        return nn.Sigmoid()
    raise KeyError(layer_type)


def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"):
    if distribution == "uniform":
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    elif distribution == "normal":
        nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        raise ValueError(distribution)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if getattr(module, "weight", None) is not None:
        nn.init.constant_(module.weight, val)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif distribution == "normal":
        nn.init.xavier_normal_(module.weight, gain=gain)
    else:
        raise ValueError(distribution)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """mmcv ConvModule with the default order conv, norm, act."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super().__init__()
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.with_spectral_norm = with_spectral_norm
        if bias == "auto":
            bias = not self.with_norm
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        if self.with_norm:
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        if self.with_activation:
            act_cfg = act_cfg.copy()
            if act_cfg["type"] not in ("Tanh", "PReLU", "Sigmoid", "HSigmoid", "Swish"):
                act_cfg.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("not used by M2MRF-C")


class NonLocal2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class ContextBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class Scale(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
