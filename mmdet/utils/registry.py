import inspect

import mmcv


class Registry(object):

    def __init__(self, name):       # 此处的self，是个对象（Object），是当前类的实例，name即为传进来的'detector'值
        self._name = name
        self._module_dict = dict()  # 定义的属性，是一个字典

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property                       # 把方法变成属性，在类外部，通过obj.name 就能获得name的值
    def name(self):
        return self._name           # 因为没有定义它的setter方法，所以是个只读属性，不能通过 self.name = newname进行修改

    @property                       # 把方法变成属性，在类外部，通过obj.module_dict 就能获得module的属性的dictionary
    def module_dict(self):
        return self._module_dict

    def get(self, key):             # 获取module属性字典中指定key的value。_module_dict是类的属性，类型是字典dict。
        return self._module_dict.get(key, None)

    # key key key function！ 注册module的关键步骤！ 在self._modult_dict中注册登记。
    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:       # 看该类是否已经登记在属性_module_dict中
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class    # 在module中dict新增key和value。key为类名，value为类对象

def register_module(self, cls):
        self._register_module(cls)
        return cls


# build_from_cfg()方法的作用是从 congfig/py配置文件中获取字典数据，
# 创建module（其实也就是一个class类），
# 然后将这个module添加到之前创建的注册表Registry的属性_module_dict中
# （这是一个字典，key为类名，value为具体的类），返回值是一个实例化后的类对象。

def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        ### 这个cfg就是py配置文件中的字典。
        # 在py配置文件中，基本上dict都会有一个key为"type"。
        # 这里，我们主要讲的是注册表DETECTORS，所以此时cfg对应的是配置文件中的model的dict{}。
        # 举个例子：比如type='CascadeRCNN'，后面我们会知道，这个value为"CascadeRCNN"的，其实就是models文件夹中某py文件中的类名，
        # 他们通过@DETECTORS.register_module，将类名当做形参，传入register_module。并保存下来。

        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')       # 字典的pop作用：移除序列中key为‘type’的元素，并且返回该元素的值
    if mmcv.is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:           # 如果obj_type已经注册到注册表registry中，即在属性_module_dict中，则obj_type 不为None
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)     # 将default_args的键值对加入到args中，将模型和训练配置进行整合，然后送入类中返回
    return obj_cls(**args)
