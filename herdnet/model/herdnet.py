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
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            layers = list(base_model.children())[:-2]  # quitar avgpool y fc
            self.encoder = nn.Sequential(*layers)
            in_channels = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            layers = list(base_model.children())[:-2]  # quitar avgpool y fc
            self.encoder = nn.Sequential(*layers)
            in_channels = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone} no soportado aún")

        # 2. Decoder / Head: generación del mapa de densidad con upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # upsample para igualar tamaño de entrada
            # nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            
            # nn.Conv2d(512, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            
            # nn.Conv2d(256, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            
            # nn.Conv2d(128, num_classes, kernel_size=1),

            # nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x_in = x
        x = self.encoder(x)
        x = self.decoder(x)
        # Recortar si hay diferencia por divisibilidad
        if x.shape[2:] != x_in.shape[2:]:
            x = x[:, :, :x_in.shape[2], :x_in.shape[3]]
        return x  # output: density map [B, num_classes, H, W]
    
    def predict(self, x, threshold=0.3):
        """
        Extrae detecciones como puntos (x, y, class_id)
        """
        self.eval()
        with torch.no_grad():
            density_maps = self.forward(x)  # [B, num_classes, H, W]
            
            batch_detections = []
            for b in range(x.shape[0]):
                detections = []
                for class_id in range(self.num_classes):
                    # Extraer puntos de cada canal (clase)
                    density = density_maps[b, class_id]  # [H, W]
                    
                    # Encontrar máximos locales
                    points = self._find_local_maxima(density, threshold)
                    
                    for y, x in zip(*points):
                        detections.append((x.item(), y.item(), class_id))
                
                batch_detections.append(detections)
            
            return batch_detections

def build_herdnet(config):
    model_config = config['model']
    model = HerdNet(
        backbone=model_config.get('backbone', 'resnet18'),
        pretrained=model_config.get('pretrained', True),
        output_stride=model_config.get('output_stride', 8),
        num_classes=config['dataset']['num_classes']
    )
    return model
