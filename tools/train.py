from __future__ import division
import argparse  # python标准库里面用来处理****命令行参数****的库
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


# 获取命令行参数的一个过程，从创建解析对象，到对其解析
def parse_args():
    # argparse是python标准库里面用来处理****命令行参数****的库
    # ArgumentParser() --参数须知：一般我们只选择用description
    # description=,    - help时显示的开始文字
    parser = argparse.ArgumentParser(description='Train a detector')   # 创建一个解析器对象, argparse库 里面的 ArgumentParser类 的对象

    # add_argument() --向该对象中添加你要关注的命令行参数和选项
    # help=,	- 写帮助信息
    # action=,	 -表示值赋予键的方式，这里用到的是bool类型 
    # type=,    - 指定参数类型
    
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    
    # 进行解析。返回值是一个命名空间，包含传递给命令的参数。
    args = parser.parse_args()   
    
    if 'LOCAL_RANK' not in os.environ:    # os.environ['']: 获取系统环境变量
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

# 函数主入口。
# 首先，做了一些config文件，work_dir以及log的操作（这些操作都是从命令行获得的，或者从命令行带有的文件里得到的参数等。）
# 然后，最主要的三个步骤：
# 1)调用build_detector() -创建模型；
# 2)调用build_dataset() -对数据集进行注册；
# 3)调用train_detector() -训练检测器。

def main():
    '''
    STEP 1 : 预备，读入配置。
    '''
    # 从命令行中读取输入参数，并记录在命名空间args中。
    args = parse_args()

    # args指定的config file中读取配置，存储在cfg中。
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    '''
    设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    '''
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # update configs according to CLI(command line interface) args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    '''
    STEP 2 : 建立模型。实质是去注册，通过调用Registry类。
    '''
    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    '''
    STEP 3 : 打包整理数据，并训练模型。
    '''
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in checkpoints as meta data. 保存数据的概况。meta data: 'data about data'.
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # 训练模型
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()


### 执行训练的命令示例： python tools/train.py configs/faster_rcnn_r50_fpn_1x.py 
#  所以，build_detection()就是将py配置文件里的数据，加载到建立的模型中，
#  然后，根据py配置文件中的数据集路径，执行build_dataset()加载数据集模型，
#  最后，进行训练train_detector()。  