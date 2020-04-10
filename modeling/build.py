# -*- coding: utf-8 -*-

from detectron2.config import CfgNode
from detectron2.modeling import build_model as detectron2_build_model


def build_model(cfg: CfgNode):
    return detectron2_build_model(cfg)

