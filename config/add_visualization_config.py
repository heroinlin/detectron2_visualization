# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_visualization_config(cfg: CN):
    _C = cfg

    _C.VISUALIZATION = CN()
    _C.VISUALIZATION.NAME = "guided_grad_cam"
    _C.VISUALIZATION.TARGET_LAYER = 1
    _C.VISUALIZATION.CLASS_INDEX = 3

    cfg.MODEL.DEVICE = "cpu"
    _C.MODEL.OSNET = CN()
    _C.MODEL.OSNET.LAYERS = [2, 2, 2]
    _C.MODEL.OSNET.CHANNELS = [32, 128, 192, 256]
    _C.MODEL.OSNET.FC_LAYER_DIM = 512
    _C.MODEL.OSNET.CLASS_NUM = 6629

    _C.PRE_PROCESS = CN()
    _C.PRE_PROCESS.MEAN = [0.44594265, 0.43469134, 0.43260086]
    _C.PRE_PROCESS.STD = [0.25284901, 0.24816869, 0.24989745]
    _C.PRE_PROCESS.WIDTH = 144
    _C.PRE_PROCESS.HEIGHT = 144
