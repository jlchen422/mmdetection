from torch import nn

from mmdet.utils import build_from_cfg
from .registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                       ROI_EXTRACTORS, SHARED_HEADS)

'''
# build_xxx()关键是调用了build()函数，而它又调用了build_from_cfg(cfg, registry, default_args).
# 其中 （参数1 -- cfg）是 （参数2 -- registry类实例） 在 config配置文件中对应 的参数。
# 比如说， 当参数2 是 DETECTORS， 那么参数1-cfg 就是在config里DETECTOR相关的配置参数。
'''
def build(cfg, registry, default_args=None):
    ##  主干是一个判断结构，其实就是判断传进来的cfg是字典列表还是单独的字典，来分情况处理。
    #   字典列表的话：挨个调用build_from_cfg()，将其加到注册表******的_module_dict中，然后再返回return nn.Sequential(*modules)
    #   字典的话：直接调用build_from_cfg()，将其添加到注册表DETECTORS中（以DETECTORS为例）。
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
