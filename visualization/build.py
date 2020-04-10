import torch
from detectron2.config import CfgNode
from .src import GradCam, GuidedBackprop, guided_grad_cam


def build_visualization(cfg: CfgNode, model: torch.nn.Module, image_tensor:torch.Tensor):
    """
    Build a visualization from config.
    """
    name = cfg.VISUALIZATION.NAME
    class_index = cfg.VISUALIZATION.CLASS_INDEX
    target_layer = cfg.VISUALIZATION.TARGET_LAYER
    visualization = None
    if name == "grad_cam":
        grad_cam = GradCam(model, target_layer)
        visualization = grad_cam.generate_cam(image_tensor, class_index)
    if name == "guided_grad_cam":
        grad_cam = GradCam(model, target_layer)
        cam = grad_cam.generate_cam(image_tensor, class_index)
        GBP = GuidedBackprop(model)
        guided_grads = GBP.generate_gradients(image_tensor, class_index)
        visualization = guided_grad_cam(cam, guided_grads)
    return visualization
