from data_transformer import BrandImageDataset

import os
import copy
import os.path as osp
import xml.etree.ElementTree as ET

import torch, torchvision
import mmdet
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
import mmcv
from mmcv import collect_env, Config
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import numpy as np

import argparse
import pickle

def main(argc=0, argv=None):
    # Dump environment and vesion info
    print("[INFO] collected env: {}".format(collect_env()))
    print("[INFO] Torch version: {}, CUDA available?: {}".format(torch.__version__,
                                                                 torch.cuda.is_available()))
    print("[INFO] mmcv version: {}".format(mmdet.mmcv_version))
    print("[INFO] CUDA version: {}".format(get_compiling_cuda_version()))
    print("[INFO] gcc version:  {}".format(get_compiler_version()))

    DATA_ROOT = argv['data_root']

    ### Load config
    cfg = Config.fromfile(argv['config_base'])

    cfg.dataset_type = 'BrandImageDataset'
    cfg.data_root = DATA_ROOT

    cfg.data.test.type = 'BrandImageDataset'
    cfg.data.test.data_root = DATA_ROOT
    cfg.data.test.ann_file = argv['train_data']
    cfg.data.test.img_prefix = argv['images']

    cfg.data.train.type = 'BrandImageDataset'
    cfg.data.train.data_root = DATA_ROOT
    cfg.data.train.ann_file = argv['train_data']
    cfg.data.train.img_prefix = argv['images']

    cfg.data.val.type = 'BrandImageDataset'
    cfg.data.val.data_root = DATA_ROOT
    cfg.data.val.ann_file = argv['val_data']
    cfg.data.val.img_prefix = argv['images']

    # TODO: Make modular number of classes
    cfg.model.roi_head.bbox_head.num_classes = 6

    # Load pretrained model
    cfg.load_from = argv['pretrained_model']

    # Setup working dir to save files and logs
    cfg.work_dir = argv['work_dir']

    cfg.optimizer.lr = 0.02/8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.interval = 12
    cfg.checkpoint_config.interval = 12

    cfg.seed = argv['seed']
    set_random_seed(argv['seed'], deterministic=False)
    cfg.gpu_ids = range(1)

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
    
    cfg.device = argv['Device']


    ### Training

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build detector
    model = build_detector(cfg.model)
    # Add attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    cfg.CLASSES = datasets[0].CLASSES

    cfg.dump(os.path.sep.join([argv['work_dir'], 'train_cfg.py']))
    print("[INFO] Training config: {}".format(cfg.pretty_text))

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)

    with open(os.path.sep.join([cfg.work_dir, 'model.pkl']), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.sep.join([cfg.work_dir, 'config.pkl']), 'wb') as f:
        pickle.dump(cfg, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train brand logo detection model.')
    parser.add_argument('-d', '--data_root', required=True,
                        help="path to data directory root")
    parser.add_argument('-c', '--config_base',
                        default='../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py',
                        help="path to base mmdet config to use")
    parser.add_argument('-t', '--train_data', default='train.txt',
                        help="path to train split file within data root directory")
    parser.add_argument('-v', '--val_data', default='val.txt',
                        help="path to val split file within data root directory")
    parser.add_argument('-i', '--images', default='images',
                        help="path to images directory within data root directory")
    parser.add_argument('-p', '--pretrained_model',
                        default='../checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth',
                        help="path to pretrained mmdet model checkpoint to load weights from")
    parser.add_argument('-w', '--work_dir', default='../brand_exps',
                        help="directory to save files and logs")
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help="pseudorandom number generator seed to use, default 0")
    parser.add_argument('-D', '--Device', default='cuda',
                        help="device to use for training, default cuda")
    argv = vars(parser.parse_args())
    main(len(argv), argv)
