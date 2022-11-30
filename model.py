import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152, ResNet152_Weights, vgg19, VGG19_Weights, vit_b_16, ViT_B_16_Weights, resnet50, ResNet50_Weights
from fastai.vision.models.unet import DynamicUnet

import model

nclasses = 20 

class ResNet(nn.Module):
    def __init__(self):
        import copy
        super(ResNet, self).__init__()
        m = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.layers = copy.deepcopy(nn.Sequential(*list(m.children())[:-2]))
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(8192, nclasses)
        )
        del m

    def forward(self, x):
        x = self.layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        pretrained = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        layers = []
        layers_linear = []
        child_counter = 0
        sub_child_counter = 0
        for child in pretrained.children():
            if child_counter == 2:
                for _child in child:
                    layers_linear.append(_child)
                    sub_child_counter += 1
                    if sub_child_counter >= 6:
                        break
            else:
                layers.append(child)
            child_counter += 1
        self.layers = nn.Sequential(*layers)
        self.layers_linear = nn.Sequential(*layers_linear)

        self.classifier = nn.Sequential(
            nn.Linear(4096, nclasses, bias=True)
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.layers_linear(x)
        x = self.classifier(x)
        return x

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.layers = vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.layers.heads = nn.Sequential(nn.Linear(768, nclasses))
    def forward(self, x):
        x = self.layers(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m = nn.Sequential(*list(m.children())[:-2])
        self.layers = DynamicUnet(m, 3, (128, 128), norm_type=None)
        self.encoder = list(list(self.layers.children())[0].children())[0]

    def forward(self, x):
        x = self.layers(x)
        return x

class resnet_ssl(nn.Module):
    def __init__(self):
        super(resnet_ssl, self).__init__()
        # m = UNET()
        m = resnet50()
        m = nn.Sequential(*list(m.children())[:-2])
        state_dict = torch.load("experiment" + '/ssl_last_model' + '.pth')
        m.load_state_dict(state_dict)
        self.layers = m
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(2,2))
        self.classifier = nn.Sequential(
            nn.Linear(8192, nclasses)
        )
    def forward(self, x):
        x = self.layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m = nn.Sequential(*list(m.children())[:-2])
        self.layers = m
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(2,2))
        self.classifier = nn.Sequential(
            nn.Linear(8192, nclasses)
        )
    def forward(self, x):
        x = self.layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class deeplab(nn.Module):
    def __init__(self):
        import copy
        super(deeplab, self).__init__()
        m = deeplabv3_resnet101(weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        self.layers = copy.deepcopy(nn.Sequential(*list(m.backbone.children())))
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(8192, nclasses)
        )
    def forward(self, x):
        x = self.layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    pass