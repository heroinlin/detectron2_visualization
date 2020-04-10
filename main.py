import argparse

import torch
from detectron2.config import get_cfg

from config import add_visualization_config
from visualization import build_visualization
from modeling import OSNet, parser_feature_and_classifier_layers
from pre_process import build_pre_process
from PIL import Image
import numpy as np
import cv2


def setup(config_file_path: str):
    cfg = get_cfg()
    add_visualization_config(cfg)
    if config_file_path is not None and len(config_file_path):
        cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg


def argument_parser():
    parser = argparse.ArgumentParser(
        description="feature visualization")
    parser.add_argument("--config_file",
                        default="configs/osnet.yaml",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--image_path",
                        default="data/1.jpg",
                        metavar="FILE",
                        help="path to image file")
    args = parser.parse_args()
    return args


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def load_checkpoint(model, checkpoint_path):
    check_point = torch.load(checkpoint_path)
    key_list = []
    mapped_state_dict = model.state_dict()
    for key, value in mapped_state_dict.items():
        key_list.append(key)
    for index, (key, value) in enumerate(check_point['net'].items()):
        # print("load ", index, key, key_list[index])
        mapped_state_dict[key_list[index]] = value
    model.load_state_dict(mapped_state_dict)
    return model


def main():
    args = argument_parser()
    config_file_path = args.config_file
    cfg = setup(config_file_path)
    image_path = args.image_path
    image = cv2.imread(image_path)
    image_tensor = build_pre_process(cfg, image)
    model = OSNet(cfg)
    checkpoint_path = r"checkpoints\osnet_x0_5_62_1.6187.pth"
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    model.features = parser_feature_and_classifier_layers(model)

    visualization = build_visualization(cfg=cfg, model=model, image_tensor=image_tensor)
    visualization = visualization - visualization.min()
    visualization /= visualization.max()
    if isinstance(visualization, (np.ndarray, np.generic)):
        visualization = format_np_output(visualization)
        visualization = Image.fromarray(visualization)
    visualization.save("1.jpg")


if __name__ == "__main__":
    main()
