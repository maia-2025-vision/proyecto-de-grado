import torch
import torch.nn as nn
from torchvision import models

class HerdNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, output_stride=8, num_classes=1):
        super().__init__()

        # 1. Cargar backbone
        self.backbone_name = backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            layers = list(base_model.children())[:-2]  # quitar avgpool y fc
            self.encoder = nn.Sequential(*layers)
            in_channels = 512
        else:
            raise NotImplementedError(f"Backbone {backbone} no soportado aún")

        # 2. Decoder / Head: generación del mapa de densidad
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # output: density map [B, num_classes, H', W']

def build_herdnet(config):
    model_config = config['model']
    model = HerdNet(
        backbone=model_config.get('backbone', 'resnet18'),
        pretrained=model_config.get('pretrained', True),
        output_stride=model_config.get('output_stride', 8),
        num_classes=config['dataset']['num_classes']
    )
    return model
