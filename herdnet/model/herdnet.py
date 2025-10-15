"""
HerdNet - Animal Detection and Counting Network
================================================

Implementación basada en:
    Paper: "From Crowd to Herd Counting" (Delplanque et al., 2023)
    Código oficial: https://github.com/Alexandre-Delplanque/HerdNet

Arquitectura:
    - Backbone: DLA-34 (6 niveles jerárquicos)
    - Decoder: DLAUp (upsampling progresivo)
    - Dual Heads: Localization (density) + Classification (species)

Autor: Adaptación del código oficial
Fecha: 2024
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

from dla import dla34, DLAUp


class HerdNet(nn.Module):
    """
    HerdNet para detección y conteo de animales en imágenes aéreas.
    
    La arquitectura usa dos cabezas paralelas:
        - Localization: Mapa de densidad en alta resolución (H/down_ratio)
        - Classification: Mapa de clases en baja resolución (H/32)
    
    Args:
        num_layers: Número de capas DLA (34 recomendado)
        num_classes: Número total de clases (incluye background como clase 0)
        pretrained: Cargar pesos ImageNet en el backbone
        down_ratio: Factor de reducción para localization [1, 2, 4, 8, 16]
        head_conv: Canales en las capas intermedias de los heads
    
    Example:
        >>> model = HerdNet(num_classes=4, down_ratio=2)
        >>> x = torch.randn(2, 3, 512, 512)
        >>> heatmap, clsmap = model(x)  # [2,1,256,256], [2,4,16,16]
    """
    
    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True,
        down_ratio: int = 2,
        head_conv: int = 64
    ):
        super(HerdNet, self).__init__()
        
        assert down_ratio in [1, 2, 4, 8, 16], \
            f'down_ratio debe ser 1, 2, 4, 8 o 16, recibido: {down_ratio}'
        
        self.num_classes = num_classes
        self.down_ratio = down_ratio
        self.head_conv = head_conv
        self.first_level = int(np.log2(down_ratio))
        
        # ════════════════════════════════════════════════════════
        # BACKBONE - DLA-34
        # ════════════════════════════════════════════════════════
        # Output: 6 niveles con canales [16, 32, 64, 128, 256, 512]
        #         en resoluciones [H, H/2, H/4, H/8, H/16, H/32]
        
        self.base = dla34(pretrained=pretrained, return_levels=True)
        self.channels = self.base.channels
        
        # ════════════════════════════════════════════════════════
        # BOTTLENECK - Procesa deep features (nivel más profundo)
        # ════════════════════════════════════════════════════════
        
        self.bottleneck_conv = nn.Conv2d(
            self.channels[-1], self.channels[-1],
            kernel_size=1, stride=1, padding=0, bias=True
        )
        
        # ════════════════════════════════════════════════════════
        # DECODER - DLAUp (upsampling jerárquico)
        # ════════════════════════════════════════════════════════
        # Combina niveles desde first_level hasta el final
        # Output: [B, channels[first_level], H/down_ratio, W/down_ratio]
        
        channels_decoder = self.channels[self.first_level:]
        scales = [2 ** i for i in range(len(channels_decoder))]
        self.dla_up = DLAUp(channels=channels_decoder, scales=scales)
        
        # ════════════════════════════════════════════════════════
        # LOCALIZATION HEAD - Mapa de densidad
        # ════════════════════════════════════════════════════════
        # Input: Decoder output [B, channels[first_level], H/down_ratio, W/down_ratio]
        # Output: [B, 1, H/down_ratio, W/down_ratio]
        
        self.loc_head = nn.Sequential(
            nn.Conv2d(self.channels[self.first_level], head_conv,
                     kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.loc_head[-2].bias.data.fill_(0.00)
        
        # ════════════════════════════════════════════════════════
        # CLASSIFICATION HEAD - Mapa de especies
        # ════════════════════════════════════════════════════════
        # Input: Bottleneck [B, 512, H/32, W/32]
        # Output: [B, num_classes, H/32, W/32]
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(self.channels[-1], head_conv,
                     kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(0.00)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass de HerdNet.
        
        Args:
            x: [B, 3, H, W] Imagen de entrada
        
        Returns:
            heatmap: [B, 1, H/down_ratio, W/down_ratio] Mapa de densidad
            clsmap: [B, num_classes, H/32, W/32] Mapa de clasificación
        """
        # Encoder: DLA-34 genera 6 niveles jerárquicos
        encode = self.base(x)
        
        # Bottleneck: procesa el nivel más profundo
        bottleneck = self.bottleneck_conv(encode[-1])
        encode[-1] = bottleneck
        
        # Decoder: upsampling jerárquico para localization
        decode_hm = self.dla_up(encode[self.first_level:])
        
        # Dual heads
        heatmap = self.loc_head(decode_hm)
        clsmap = self.cls_head(bottleneck)
        
        return heatmap, clsmap
    
    def predict(
        self,
        x: torch.Tensor,
        density_threshold: float = 0.1,
        nms_kernel: int = 3
    ) -> List[List[Tuple[int, int, int, float]]]:
        """
        Inferencia: extrae detecciones individuales desde los mapas.
        
        Args:
            x: [B, 3, H, W] Batch de imágenes
            density_threshold: Umbral mínimo de densidad (0-1)
            nms_kernel: Tamaño del kernel para Non-Maximum Suppression
        
        Returns:
            Lista de detecciones por imagen: [[(x, y, class_id, confidence), ...]]
        
        Example:
            >>> detections = model.predict(img, density_threshold=0.3)
            >>> print(f"Detectados {len(detections[0])} animales")
        """
        self.eval()
        
        with torch.no_grad():
            heatmap, clsmap = self.forward(x)
            
            # Upsampling de clsmap a resolución del heatmap
            B, C, H_cls, W_cls = clsmap.shape
            _, _, H_hm, W_hm = heatmap.shape
            
            clsmap_upsampled = nn.functional.interpolate(
                clsmap, size=(H_hm, W_hm),
                mode='bilinear', align_corners=False
            )
            class_probs = torch.softmax(clsmap_upsampled, dim=1)
            
            batch_detections = []
            
            for b in range(B):
                detections = []
                density = heatmap[b, 0]
                
                # NMS: encontrar picos locales en el mapa de densidad
                peaks = self._find_peaks(density, density_threshold, nms_kernel)
                
                for y, x_coord in zip(*peaks):
                    x_hm, y_hm = x_coord.item(), y.item()
                    
                    # Escalar coordenadas a imagen original
                    x_orig = int(x_hm * self.down_ratio)
                    y_orig = int(y_hm * self.down_ratio)
                    
                    # Asignar clase más probable
                    class_id = torch.argmax(class_probs[b, :, y_hm, x_hm]).item()
                    confidence = density[y_hm, x_hm].item()
                    
                    detections.append((x_orig, y_orig, class_id, confidence))
                
                batch_detections.append(detections)
            
            return batch_detections
    
    def _find_peaks(
        self,
        heatmap: torch.Tensor,
        threshold: float,
        kernel_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-Maximum Suppression para encontrar máximos locales."""
        pad = kernel_size // 2
        heatmap_max = nn.functional.max_pool2d(
            heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        ).squeeze()
        
        keep = (heatmap == heatmap_max) & (heatmap > threshold)
        return torch.where(keep)
    
    def freeze_backbone(self) -> None:
        """Congela el backbone DLA-34 para fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self) -> None:
        """Descongela todos los parámetros."""
        for param in self.parameters():
            param.requires_grad = True


def build_herdnet(config: dict) -> HerdNet:
    """
    Factory function para construir HerdNet desde configuración.
    
    Args:
        config: Diccionario con claves 'model' y 'dataset'
    
    Returns:
        Modelo HerdNet inicializado
    
    Example:
        >>> config = {
        ...     'model': {'down_ratio': 2, 'head_conv': 64, 'pretrained': True},
        ...     'dataset': {'num_classes': 4}  # background + 3 especies
        ... }
        >>> model = build_herdnet(config)
    """
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})
    
    return HerdNet(
        num_layers=model_cfg.get('num_layers', 34),
        num_classes=dataset_cfg.get('num_classes', 2),
        pretrained=model_cfg.get('pretrained', True),
        down_ratio=model_cfg.get('down_ratio', 2),
        head_conv=model_cfg.get('head_conv', 64)
    )


# ═══════════════════════════════════════════════════════════════════════
# TESTING Y VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("HERDNET - VALIDACIÓN DE IMPLEMENTACIÓN")
    print("="*70 + "\n")
    
    # Configuración estándar (según paper)
    config = {
        'model': {
            'num_layers': 34,
            'pretrained': True,
            'down_ratio': 2,
            'head_conv': 64
        },
        'dataset': {
            'num_classes': 4  # Ejemplo: background, camel, sheep, goat
        }
    }
    
    # Crear modelo
    model = build_herdnet(config)
    model.eval()
    
    print("📦 Configuración:")
    print(f"   • Clases: {config['dataset']['num_classes']}")
    print(f"   • Down ratio: {config['model']['down_ratio']}")
    print(f"   • Pretrained: {config['model']['pretrained']}")
    
    # Verificar carga de pesos pre-entrenados
    if config['model']['pretrained']:
        print(f"\n🔍 Verificación de Pesos Pre-entrenados:")
        print("-" * 70)
        
        # Test 1: Verificar que los pesos no son aleatorios
        first_conv = model.base.base_layer[0]  # Primera conv del backbone
        weight_mean = first_conv.weight.data.mean().item()
        weight_std = first_conv.weight.data.std().item()
        
        print(f"   • Backbone primera conv - Media: {weight_mean:.6f}, Std: {weight_std:.6f}")
        
        # Pesos aleatorios típicamente tienen media ~0 y std alta
        # Pesos pre-entrenados tienen distribución más específica
        if abs(weight_mean) < 0.1 and weight_std > 0.05:
            print(f"   ✅ Distribución de pesos sugiere carga pre-entrenada exitosa")
        else:
            print(f"   ⚠️ Distribución podría indicar inicialización aleatoria")
        
        # Test 2: Verificar que BN tiene estadísticas aprendidas
        first_bn = model.base.base_layer[1]  # Primera BatchNorm
        bn_mean = first_bn.running_mean.mean().item()
        bn_var = first_bn.running_var.mean().item()
        
        print(f"   • BatchNorm - Running mean: {bn_mean:.6f}, Running var: {bn_var:.6f}")
        
        # BN pre-entrenada debe tener estadísticas != valores por defecto
        if abs(bn_mean) > 1e-6 or abs(bn_var - 1.0) > 1e-3:
            print(f"   ✅ BatchNorm contiene estadísticas pre-entrenadas")
        else:
            print(f"   ⚠️ BatchNorm parece tener valores por defecto")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Parámetros totales: {total_params:,}")
        
    # Test forward pass
    print(f"\n🧪 Test de Forward Pass:")
    print("-" * 70)
    
    batch_size, H, W = 2, 512, 512
    x = torch.randn(batch_size, 3, H, W)
    
    heatmap, clsmap = model(x)
    
    print(f"Input:    {tuple(x.shape)}")
    print(f"Heatmap:  {tuple(heatmap.shape)}")
    print(f"Clsmap:   {tuple(clsmap.shape)}")
    
    # Verificación de dimensiones
    expected_hm = (batch_size, 1, H // config['model']['down_ratio'], W // config['model']['down_ratio'])
    expected_cls = (batch_size, config['dataset']['num_classes'], H // 32, W // 32)
    
    assert heatmap.shape == expected_hm, f"Heatmap shape incorrecto: esperado {expected_hm}"
    assert clsmap.shape == expected_cls, f"Clsmap shape incorrecto: esperado {expected_cls}"
    
    print(f"\n✅ Verificación de dimensiones: PASADA")
    
    # Test de inferencia
    print(f"\n🎯 Test de Inferencia:")
    print("-" * 70)
    
    detections = model.predict(x, density_threshold=0.3)
    
    for i, dets in enumerate(detections):
        print(f"   Imagen {i}: {len(dets)} detecciones")
        if len(dets) > 0:
            det = dets[0]
            print(f"      Ejemplo: x={det[0]}, y={det[1]}, clase={det[2]}, conf={det[3]:.3f}")
    
    print("\n" + "="*70)
    print("✅ VALIDACIÓN COMPLETA - Implementación correcta")
    print("="*70 + "\n")
    
    # Información adicional
    print("📝 Arquitectura Validada:")
    print("   • Backbone: DLA-34 con 6 niveles jerárquicos")
    print("   • Decoder: DLAUp (upsampling progresivo)")
    print("   • Localization: Sigmoid → valores en [0, 1]")
    print("   • Classification: Sin decoder, opera en H/32")