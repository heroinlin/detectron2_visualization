import argparse
import os
import cv2
import torch
from detectron2.config import get_cfg

from config import add_visualization_config
from visualization import build_visualization
from modeling import OSNet, parser_feature_and_classifier_layers
from pre_process import build_pre_process
import numpy as np
from utils.misc_functions import save_gradient_images, convert_to_grayscale


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
    parser.add_argument("--checkpoint_path",
                        default="checkpoints/osnet_x0_5_62_1.6187.pth",
                        metavar="FILE",
                        help="path to model checkpoint file")
    parser.add_argument("--save_folder",
                        default="./results",
                        metavar="FILE",
                        help="folder path to save file")
    args = parser.parse_args()
    return args


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


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cam = cv2.addWeighted(img, 0.8, heatmap, 0.5, 0)
    return cam


def main():
    args = argument_parser()
    config_file_path = args.config_file
    cfg = setup(config_file_path)
    image_path = args.image_path
    image = cv2.imread(image_path)
    image_tensor = build_pre_process(cfg, image)

    model = OSNet(cfg)
    model = load_checkpoint(model, args.checkpoint_path)
    model.eval()
    model.features = parser_feature_and_classifier_layers(model)

    visualization = build_visualization(cfg=cfg, model=model, image_tensor=image_tensor)

    image_name = os.path.basename(image_path)
    file_name_to_export = image_name[0:image_name.rfind('.')]
    save_gradient_images(visualization, os.path.join(args.save_folder, file_name_to_export + "_GGrad_Cam.jpg"))
    gray_visualization = convert_to_grayscale(visualization)
    save_gradient_images(gray_visualization, os.path.join(args.save_folder, file_name_to_export + "_GGrad_Cam_gray.jpg"))
    gray_visualization = gray_visualization.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cam_on_src = show_cam_on_image(image, gray_visualization)
    cv2.imwrite(os.path.join(args.save_folder, file_name_to_export + "_GGrad_Cam_On_src.jpg"), cam_on_src)


if __name__ == "__main__":
    main()
