import torch
import torch.nn as nn
from torchvision import models


class HerdNet(nn.Module):
    """
    HerdNet según el diagrama arquitectónico original.
    
    Características:
    - Backbone: ResNet (o DLA-34 en el paper)
    - Decoder compartido con upsampling
    - Localization Head: Mapa de alta resolución [1, H, W]
    - Classification Head: Mapa de baja resolución [C, H/16, W/16]
    """
    
    def __init__(self, backbone='resnet18', pretrained=True, num_classes=6):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # ═══════════════════════════════════════════════════
        # 1. ENCODER (Backbone)
        # ═══════════════════════════════════════════════════
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            in_channels = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            in_channels = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            in_channels = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone} no soportado")
        
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        # Output: [B, in_channels, H/32, W/32]
        
        # ═══════════════════════════════════════════════════
        # 2. DECODER COMPARTIDO (Deep Features)
        # ═══════════════════════════════════════════════════
        # Procesa features pero sin upsampling completo todavía
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample moderado: H/32 → H/16 (x2)
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Output: [B, 64, H/16, W/16]
        
        # ═══════════════════════════════════════════════════
        # 3. LOCALIZATION HEAD (Alta resolución)
        # ═══════════════════════════════════════════════════
        # Según diagrama: 3x3 → 1x1 → Output [1, 256, 256]
        self.localization_head = nn.Sequential(
            # Procesar features
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Reducir a 1 canal
            nn.Conv2d(32, 1, kernel_size=1),
            
            # Upsampling agresivo: H/16 → H (x16)
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
            
            # Activación para densidad (valores positivos)
            nn.ReLU()
        )
        # Output: [B, 1, H, W]
        
        # ═══════════════════════════════════════════════════
        # 4. CLASSIFICATION HEAD (Baja resolución)
        # ═══════════════════════════════════════════════════
        # Según diagrama: 3x3 → 1x1 → Output [C, 16, 16]
        self.classification_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, kernel_size=1),
            # NO upsampling aquí - se queda en H/16, W/16
        )
        # Output: [B, num_classes, H/16, W/16]
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] imagen de entrada
        
        Returns:
            localization_map: [B, 1, H, W] - mapa de densidad en alta resolución
            classification_maps: [B, num_classes, H/16, W/16] - mapas de clase en baja resolución
        """
        batch_size, _, H, W = x.shape
        
        # Encoder: extraer features
        features = self.encoder(x)  # [B, in_channels, H/32, W/32]
        
        # Decoder: procesar features
        decoded = self.decoder(features)  # [B, 64, H/16, W/16]
        
        # Dos cabezas con diferentes resoluciones
        localization_map = self.localization_head(decoded)     # [B, 1, H, W]
        classification_maps = self.classification_head(decoded) # [B, num_classes, H/16, W/16]
        
        # Ajustar tamaño de localization si es necesario
        if localization_map.shape[2:] != (H,W):
            localization_map = localization_map[:, :, :H, :W]
                
        return localization_map, classification_maps
    
    def predict(self, x, density_threshold=0.3):
        """
        Modo inferencia: extrae detecciones como puntos (x, y, class_id)
        
        Args:
            x: [B, 3, H, W] imagen
            density_threshold: umbral para considerar una detección
        
        Returns:
            Lista de detecciones por imagen: [(x, y, class_id, confidence), ...]
        """
        self.eval()
        with torch.no_grad():
            localization_map, classification_maps = self.forward(x)
            
            # Upsampling de classification para que coincida con localization
            B, C, H_cls, W_cls = classification_maps.shape
            H_loc, W_loc = localization_map.shape[2:]
            
            # Interpolar classification a la misma resolución que localization
            classification_upsampled = nn.functional.interpolate(
                classification_maps, 
                size=(H_loc, W_loc), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Softmax para probabilidades
            class_probs = torch.softmax(classification_upsampled, dim=1)
            
            batch_detections = []
            for b in range(x.shape[0]):
                detections = []
                density = localization_map[b, 0]  # [H, W]
                
                # Encontrar picos locales
                local_max = self._find_local_maxima(density, threshold=density_threshold)
                
                # Para cada detección, asignar clase
                for y, x_coord in zip(*local_max):
                    class_probs_pixel = class_probs[b, :, y, x_coord]
                    class_id = torch.argmax(class_probs_pixel).item()
                    confidence = density[y, x_coord].item()
                    
                    detections.append((x_coord.item(), y.item(), class_id, confidence))
                
                batch_detections.append(detections)
            
            return batch_detections
    
    def _find_local_maxima(self, heatmap, threshold=0.3, kernel_size=3):
        """
        Encuentra máximos locales en el mapa de calor.
        """
        pad = kernel_size // 2
        heatmap_max = nn.functional.max_pool2d(
            heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        ).squeeze()
        
        local_maxima = (heatmap == heatmap_max) & (heatmap > threshold)
        return torch.where(local_maxima)


def build_herdnet(config):
    """
    Construye HerdNet desde configuración.
    """
    model_config = config['model']
    
    model = HerdNet(
        backbone=model_config.get('backbone', 'resnet18'),
        pretrained=model_config.get('pretrained', True),
        num_classes=config['dataset']['num_classes']
    )
    
    return model


if __name__ == "__main__":
    config = {
        'model': {'backbone': 'resnet18', 'pretrained': True},
        'dataset': {'num_classes': 6}
    }
    
    model = build_herdnet(config)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 2000, 2000)
    localization, classification = model(dummy_input)
    
    print(f"Input shape:          {dummy_input.shape}")
    print(f"Localization shape:   {localization.shape}")     # [2, 1, 512, 512]
    print(f"Classification shape: {classification.shape}")   # [2, 6, 32, 32]
    print(f"\nNota: Localization está en ALTA resolución")
    print(f"      Classification está en BAJA resolución (H/16)")
    
    # Test inferencia
    detections = model.predict(dummy_input)
    print(f"\nDetecciones: {len(detections[0])} objetos en imagen 0")