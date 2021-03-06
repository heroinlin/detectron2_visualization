import torch
import torch.nn as nn


def parser_feature(model, name=(), features=[], features_name=[]):
    """
    提取网络特征层部分，此处特征层是分类层之前的所有网络层
    Parameters
    ----------
    model 完整网络
    name 网络层初始前缀名
    features 已提取的特征层列表
    features_name  已提取的特征层名称列表

    Returns
    -------
        特征层列表, 特征层名称列表
    """
    for name1, module1 in model._modules.items():
        # print(module1)
        name1 = name.__add__((name1,))
        # print(type(module1))
        if "classifier" in name1:
            continue
        elif isinstance(module1, nn.AdaptiveAvgPool2d):
            features.append(module1)
            features_name.append('.'.join(name1))
        elif isinstance(module1, nn.AvgPool2d):
            features.append(module1)
            features_name.append('.'.join(name1))
        elif isinstance(module1, nn.Conv2d):
            features.append(module1)
            features_name.append('.'.join(name1))
        elif isinstance(module1, nn.BatchNorm2d):
            features.append(module1)
            features_name.append('.'.join(name1))
        elif isinstance(module1, nn.ReLU):
            features.append(module1)
            features_name.append('.'.join(name1))
        elif isinstance(module1, nn.MaxPool2d):
            features.append(module1)
            features_name.append('.'.join(name1))
        elif isinstance(module1, nn.Sequential):
            features, features_name = parser_feature(module1, name1, features, features_name)
        elif len(module1._modules) and "res" not in '.'.join(name1):
            features, features_name = parser_feature(module1, name1, features, features_name)
        else:
            features.append(module1)
            features_name.append('.'.join(name1))
    return features, features_name


def parser_feature_and_classifier_layers(model):
    # 确保第一层为单个conv，而不是conv+bn的组合，否则guide_bp时会有尺寸问题
    features, features_name = parser_feature(model)
    features = torch.nn.Sequential(*features)
    print("features layers: ", features_name)
    return features
