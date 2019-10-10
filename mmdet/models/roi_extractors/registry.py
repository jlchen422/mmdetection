from mmdet.utils import Registry

# 类的实例化，Registry是一个类，传入的是一个字符串。该字符串为Registry类的name属性值
# 这些类实例下的_module_dict属性，则是用来存对应的相同类对象的，
# 举个例子：比如DETECTORS的_module_dict下就有可能有：Faster R-CNN、Cascade R-CNN、FPN、HTC等常见的检测器

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
