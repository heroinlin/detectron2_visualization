# -*- coding: utf-8 -*-

import numpy as np
import cv2
import torch
from detectron2.config import CfgNode


def build_pre_process(cfg: CfgNode, image):
    means = cfg.PRE_PROCESS.MEAN
    stds = cfg.PRE_PROCESS.STD
    width = cfg.PRE_PROCESS.WIDTH
    height = cfg.PRE_PROCESS.HEIGHT

    prep_img = cv2.resize(image, (width, height))
    prep_img = np.float32(prep_img) / 255
    # ensure or transform incoming image to PIL image
    preprocessed_img = prep_img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    im_as_ten = torch.from_numpy(preprocessed_img)
    im_as_ten.unsqueeze_(0)
    im_as_ten = im_as_ten.requires_grad_(True)
    return im_as_ten
