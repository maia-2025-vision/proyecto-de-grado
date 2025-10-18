"""
COCO Patcher - Patch Generation for HerdNet
===========================================

Implementación de generación de patches desde imágenes COCO para HerdNet.
Inspirado en la filosofía del autor original pero adaptado para formato COCO.

Funcionalidades:
    - Sliding window con overlap configurable
    - Filtrado por visibilidad mínima de objetos
    - Transformación correcta de coordenadas COCO
    - Padding inteligente para patches en bordes
    - Tests completos de validación

Autor: Adaptación para proyecto MAIA 2025
Fecha: Octubre 2025
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import cv2
from tqdm import tqdm
import unittest
import tempfile
import shutil


@dataclass
class PatchInfo:
    """Información de un patch extraído."""
    patch_id: str
    original_image_id: int
    original_filename: str
    patch_filename: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    padded: bool = False


@dataclass
class PatchStats:
    """Estadísticas del proceso de patcheo."""
    total_images: int = 0
    total_patches: int = 0
    patches_with_objects: int = 0
    total_objects: int = 0
    objects_kept: int = 0
    objects_filtered: int = 0
    
    def __str__(self):
        return f"""
Patch Generation Statistics:
===========================
Total images processed: {self.total_images}
Total patches generated: {self.total_patches}
Patches with objects: {self.patches_with_objects}
Total objects: {self.total_objects}
Objects kept: {self.objects_kept}
Objects filtered: {self.objects_filtered}
Keep ratio: {self.objects_kept/max(1,self.total_objects):.2%}
"""


class PatchExtractor:
    """
    Extrae patches de imágenes usando sliding window con overlap.
    """
    
    def __init__(self, patch_size: int, overlap: int):
        """
        Args:
            patch_size: Tamaño del patch (cuadrado)
            overlap: Overlap entre patches adyacentes
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
    def get_patch_coordinates(self, image_width: int, image_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Calcula coordenadas de todos los patches para una imagen.
        
        Returns:
            Lista de tuplas (x, y, w, h) para cada patch
        """
        patches = []
        
        # Sliding window con stride - algoritmo genérico
        y = 0
        while y < image_height:
            x = 0
            while x < image_width:
                # Determinar tamaño del patch
                patch_w = min(self.patch_size, image_width - x)
                patch_h = min(self.patch_size, image_height - y)
                
                patches.append((x, y, patch_w, patch_h))
                
                # Siguiente posición x
                if x + self.patch_size >= image_width:
                    break
                x += self.stride
            
            # Siguiente posición y  
            if y + self.patch_size >= image_height:
                break
            y += self.stride
        
        return patches
    
    def extract_patch(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[np.ndarray, bool]:
        """
        Extrae un patch de la imagen con padding si es necesario.
        
        Returns:
            Tuple[patch, was_padded]
        """
        # Extraer región
        patch = image[y:y+h, x:x+w]
        was_padded = False
        
        # Padding si el patch es menor que patch_size
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            # Crear patch con padding
            padded_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=patch.dtype)
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
            was_padded = True
            
        return patch, was_padded


class AnnotationAdjuster:
    """
    Ajusta anotaciones COCO para patches individuales.
    """
    
    def __init__(self, min_visibility: float = 0.1):
        """
        Args:
            min_visibility: Fracción mínima del área que debe ser visible
        """
        self.min_visibility = min_visibility
    
    def adjust_annotations(
        self, 
        annotations: List[Dict], 
        patch_x: int, 
        patch_y: int, 
        patch_w: int, 
        patch_h: int
    ) -> List[Dict]:
        """
        Ajusta anotaciones para un patch específico.
        
        Args:
            annotations: Lista de anotaciones COCO originales
            patch_x, patch_y: Posición del patch en imagen original
            patch_w, patch_h: Dimensiones del patch
            
        Returns:
            Lista de anotaciones ajustadas para el patch
        """
        adjusted_annotations = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height] formato COCO
            orig_x, orig_y, orig_w, orig_h = bbox
            
            # Calcular intersección con el patch
            intersect_x1 = max(orig_x, patch_x)
            intersect_y1 = max(orig_y, patch_y)
            intersect_x2 = min(orig_x + orig_w, patch_x + patch_w)
            intersect_y2 = min(orig_y + orig_h, patch_y + patch_h)
            
            # Verificar si hay intersección
            if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
                continue  # Sin intersección
            
            # Calcular área de intersección
            intersect_w = intersect_x2 - intersect_x1
            intersect_h = intersect_y2 - intersect_y1
            intersect_area = intersect_w * intersect_h
            original_area = orig_w * orig_h
            
            # Filtrar por visibilidad mínima
            visibility = intersect_area / original_area
            if visibility < self.min_visibility:
                continue
            
            # Ajustar coordenadas al patch
            new_x = intersect_x1 - patch_x
            new_y = intersect_y1 - patch_y
            new_w = intersect_w
            new_h = intersect_h
            
            # Crear nueva anotación
            adjusted_ann = ann.copy()
            adjusted_ann['bbox'] = [new_x, new_y, new_w, new_h]
            adjusted_ann['area'] = new_w * new_h
            adjusted_ann['_original_visibility'] = visibility
            
            adjusted_annotations.append(adjusted_ann)
        
        return adjusted_annotations


class VisibilityFilter:
    """
    Filtra objetos y patches basado en criterios de visibilidad.
    """
    
    def __init__(self, min_visibility: float = 0.1):
        self.min_visibility = min_visibility
    
    def should_keep_patch(self, annotations: List[Dict], save_all: bool = False) -> bool:
        """
        Determina si un patch debe ser guardado.
        
        Args:
            annotations: Anotaciones del patch
            save_all: Si True, guarda todos los patches
            
        Returns:
            True si el patch debe ser guardado
        """
        if save_all:
            return True
        return len(annotations) > 0
    
    def filter_objects_by_visibility(self, annotations: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Filtra objetos por visibilidad mínima.
        
        Returns:
            Tuple[annotations_filtradas, num_filtrados]
        """
        kept = []
        filtered_count = 0
        
        for ann in annotations:
            visibility = ann.get('_original_visibility', 1.0)
            if visibility >= self.min_visibility:
                kept.append(ann)
            else:
                filtered_count += 1
                
        return kept, filtered_count


class COCOPatcher:
    """
    Generador principal de patches desde formato COCO.
    """
    
    def __init__(
        self,
        patch_size: int = 512,
        overlap: int = 128,
        min_visibility: float = 0.1,
        save_all_patches: bool = False
    ):
        """
        Args:
            patch_size: Tamaño de patches (cuadrado)
            overlap: Overlap entre patches
            min_visibility: Visibilidad mínima para conservar objetos
            save_all_patches: Guardar patches sin objetos
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.save_all_patches = save_all_patches
        
        self.extractor = PatchExtractor(patch_size, overlap)
        self.adjuster = AnnotationAdjuster(min_visibility)
        self.filter = VisibilityFilter(min_visibility)
        
    def process_coco_dataset(
        self,
        coco_json_path: str,
        images_dir: str,
        output_dir: str,
        output_coco_path: Optional[str] = None
    ) -> PatchStats:
        """
        Procesa dataset COCO completo generando patches.
        
        Args:
            coco_json_path: Ruta al archivo COCO JSON
            images_dir: Directorio con imágenes originales
            output_dir: Directorio de salida para patches
            output_coco_path: Ruta para nuevo COCO JSON (opcional)
            
        Returns:
            Estadísticas del proceso
        """
        # Cargar COCO
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Preparar directorios
        os.makedirs(output_dir, exist_ok=True)
        
        # Estadísticas
        stats = PatchStats()
        
        # Estructura para nuevo COCO
        new_coco = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data.get('categories', []),
            'images': [],
            'annotations': []
        }
        
        # Mapas para búsqueda rápida
        image_id_to_data = {img['id']: img for img in coco_data['images']}
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        patch_id_counter = 1
        annotation_id_counter = 1
        
        # Procesar cada imagen
        for image_data in tqdm(coco_data['images'], desc="Processing images"):
            image_id = image_data['id']
            filename = image_data['file_name']
            
            # Cargar imagen
            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                continue
                
            image = np.array(Image.open(image_path))
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass  # RGB OK
            else:
                print(f"⚠️ Skipping non-RGB image: {filename}")
                continue
                
            # Obtener anotaciones de la imagen
            image_annotations = annotations_by_image.get(image_id, [])
            stats.total_objects += len(image_annotations)
            
            # Generar patches
            patch_coords = self.extractor.get_patch_coordinates(
                image_data['width'], image_data['height']
            )
            
            stats.total_images += 1
            
            for patch_x, patch_y, patch_w, patch_h in patch_coords:
                # Extraer patch
                patch_array, was_padded = self.extractor.extract_patch(
                    image, patch_x, patch_y, patch_w, patch_h
                )
                
                # Ajustar anotaciones
                patch_annotations = self.adjuster.adjust_annotations(
                    image_annotations, patch_x, patch_y, patch_w, patch_h
                )
                
                # Filtrar por visibilidad
                patch_annotations, filtered_count = self.filter.filter_objects_by_visibility(
                    patch_annotations
                )
                stats.objects_filtered += filtered_count
                stats.objects_kept += len(patch_annotations)
                
                # Decidir si guardar el patch
                if not self.filter.should_keep_patch(patch_annotations, self.save_all_patches):
                    continue
                
                # Generar nombre del patch
                base_name = os.path.splitext(filename)[0]
                patch_filename = f"{base_name}_patch_{patch_id_counter:04d}.jpg"
                patch_path = os.path.join(output_dir, patch_filename)
                
                # Guardar patch
                patch_pil = Image.fromarray(patch_array.astype(np.uint8))
                patch_pil.save(patch_path, quality=95)
                
                # Agregar imagen al nuevo COCO
                new_image_data = {
                    'id': patch_id_counter,
                    'file_name': patch_filename,
                    'width': self.patch_size,
                    'height': self.patch_size,
                    'original_image_id': image_id,
                    'original_filename': filename,
                    'patch_x': patch_x,
                    'patch_y': patch_y,
                    'was_padded': was_padded
                }
                new_coco['images'].append(new_image_data)
                
                # Agregar anotaciones al nuevo COCO
                for ann in patch_annotations:
                    new_ann = ann.copy()
                    new_ann['id'] = annotation_id_counter
                    new_ann['image_id'] = patch_id_counter
                    # Remover campos internos
                    new_ann.pop('_original_visibility', None)
                    new_coco['annotations'].append(new_ann)
                    annotation_id_counter += 1
                
                stats.total_patches += 1
                if len(patch_annotations) > 0:
                    stats.patches_with_objects += 1
                    
                patch_id_counter += 1
        
        # Guardar nuevo COCO JSON
        if output_coco_path:
            with open(output_coco_path, 'w') as f:
                json.dump(new_coco, f, indent=2)
        
        return stats


# ═══════════════════════════════════════════════════════════════════════
# TESTS DE VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════

class TestPatchExtractor(unittest.TestCase):
    """Tests para PatchExtractor."""
    
    def setUp(self):
        self.extractor = PatchExtractor(patch_size=512, overlap=128)
    
    def test_patch_coordinates_simple(self):
        """Test coordenadas para imagen que cabe en un solo patch."""
        # Para patch_size=512, usar imagen que quepa en un solo patch
        coords = self.extractor.get_patch_coordinates(400, 300)
        
        # Imagen 400x300 con patch_size=512 debe generar 1 solo patch
        self.assertEqual(len(coords), 1)
        self.assertEqual(coords[0], (0, 0, 400, 300))
    
    def test_patch_coordinates_medium(self):
        """Test coordenadas para imagen mediana que requiere múltiples patches."""
        # Imagen 600x400 con patch_size=512, stride=384 
        coords = self.extractor.get_patch_coordinates(600, 400)
        
        # Debe generar 2 patches horizontalmente
        self.assertEqual(len(coords), 2)
        self.assertEqual(coords[0], (0, 0, 512, 400))       # Primer patch
        self.assertEqual(coords[1], (384, 0, 216, 400))     # Segundo patch
    
    def test_patch_coordinates_large(self):
        """Test coordenadas para imagen grande."""
        coords = self.extractor.get_patch_coordinates(1024, 1024)
        # Con patch_size=512, stride=384, debería generar múltiples patches
        self.assertGreater(len(coords), 1)
        
        # Verificar que cubren toda la imagen
        min_x = min(x for x, _, _, _ in coords)
        max_x = max(x + w for x, _, w, _ in coords)
        min_y = min(y for _, y, _, _ in coords)  
        max_y = max(y + h for _, y, _, h in coords)
        
        self.assertEqual(min_x, 0)
        self.assertEqual(min_y, 0)
        self.assertGreaterEqual(max_x, 1024)
        self.assertGreaterEqual(max_y, 1024)
    
    def test_extract_patch_no_padding(self):
        """Test extracción sin padding."""
        image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
        patch, was_padded = self.extractor.extract_patch(image, 0, 0, 512, 512)
        
        self.assertEqual(patch.shape, (512, 512, 3))
        self.assertFalse(was_padded)
    
    def test_extract_patch_with_padding(self):
        """Test extracción con padding."""
        image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        patch, was_padded = self.extractor.extract_patch(image, 0, 0, 300, 400)
        
        self.assertEqual(patch.shape, (512, 512, 3))
        self.assertTrue(was_padded)
        # Verificar que el área original está preservada
        np.testing.assert_array_equal(patch[:300, :300], image[:300, :300])


class TestAnnotationAdjuster(unittest.TestCase):
    """Tests para AnnotationAdjuster."""
    
    def setUp(self):
        self.adjuster = AnnotationAdjuster(min_visibility=0.1)
    
    def test_annotation_completely_inside_patch(self):
        """Test objeto completamente dentro del patch."""
        annotations = [
            {
                'id': 1,
                'bbox': [100, 150, 50, 40],  # x, y, w, h
                'area': 2000,
                'category_id': 1
            }
        ]
        
        adjusted = self.adjuster.adjust_annotations(
            annotations, patch_x=50, patch_y=100, patch_w=512, patch_h=512
        )
        
        self.assertEqual(len(adjusted), 1)
        # Coordenadas ajustadas: [100-50, 150-100, 50, 40] = [50, 50, 50, 40]
        self.assertEqual(adjusted[0]['bbox'], [50, 50, 50, 40])
        self.assertEqual(adjusted[0]['_original_visibility'], 1.0)
    
    def test_annotation_partial_intersection(self):
        """Test objeto parcialmente fuera del patch."""
        annotations = [
            {
                'id': 1,
                'bbox': [480, 480, 100, 100],  # Se sale del patch 512x512
                'area': 10000,
                'category_id': 1
            }
        ]
        
        adjusted = self.adjuster.adjust_annotations(
            annotations, patch_x=0, patch_y=0, patch_w=512, patch_h=512
        )
        
        self.assertEqual(len(adjusted), 1)
        # Intersección: [480, 480, 32, 32] (solo la parte que cabe)
        self.assertEqual(adjusted[0]['bbox'], [480, 480, 32, 32])
        self.assertAlmostEqual(adjusted[0]['_original_visibility'], 0.1024, places=3)
    
    def test_annotation_no_intersection(self):
        """Test objeto fuera del patch."""
        annotations = [
            {
                'id': 1, 
                'bbox': [600, 600, 50, 50],  # Completamente fuera
                'area': 2500,
                'category_id': 1
            }
        ]
        
        adjusted = self.adjuster.adjust_annotations(
            annotations, patch_x=0, patch_y=0, patch_w=512, patch_h=512
        )
        
        self.assertEqual(len(adjusted), 0)
    
    def test_min_visibility_filtering(self):
        """Test filtrado por visibilidad mínima."""
        # Objeto con visibilidad menor al mínimo
        annotations = [
            {
                'id': 1,
                'bbox': [500, 500, 100, 100],  # Solo 12x12 visible
                'area': 10000,
                'category_id': 1
            }
        ]
        
        adjusted = self.adjuster.adjust_annotations(
            annotations, patch_x=0, patch_y=0, patch_w=512, patch_h=512
        )
        
        # Visibilidad = (12*12)/(100*100) = 0.0144 < 0.1, debería filtrarse
        self.assertEqual(len(adjusted), 0)


class TestCOCOPatcher(unittest.TestCase):
    """Tests de integración para COCOPatcher."""
    
    def setUp(self):
        self.patcher = COCOPatcher(
            patch_size=256,  # Más pequeño para tests
            overlap=64,
            min_visibility=0.1,
            save_all_patches=False
        )
        
        # Crear dataset COCO de prueba
        self.test_coco = {
            'images': [
                {
                    'id': 1,
                    'file_name': 'test_image.jpg',
                    'width': 600,
                    'height': 400
                }
            ],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'bbox': [100, 50, 80, 60],
                    'area': 4800,
                    'category_id': 1
                },
                {
                    'id': 2,
                    'image_id': 1,  
                    'bbox': [300, 200, 40, 30],
                    'area': 1200,
                    'category_id': 2
                }
            ],
            'categories': [
                {'id': 1, 'name': 'animal1'},
                {'id': 2, 'name': 'animal2'}
            ]
        }
    
    def test_end_to_end_processing(self):
        """Test completo de procesamiento."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
            image_path = os.path.join(temp_dir, 'test_image.jpg')
            Image.fromarray(test_image).save(image_path)
            
            # Crear COCO JSON de prueba
            coco_path = os.path.join(temp_dir, 'test_coco.json')
            with open(coco_path, 'w') as f:
                json.dump(self.test_coco, f)
            
            # Procesar
            output_dir = os.path.join(temp_dir, 'patches')
            output_coco_path = os.path.join(temp_dir, 'patches_coco.json')
            
            stats = self.patcher.process_coco_dataset(
                coco_path, temp_dir, output_dir, output_coco_path
            )
            
            # Verificaciones básicas
            self.assertEqual(stats.total_images, 1)
            self.assertGreater(stats.total_patches, 0)
            self.assertEqual(stats.total_objects, 2)
            self.assertTrue(os.path.exists(output_coco_path))
            
            # Verificar nuevo COCO
            with open(output_coco_path, 'r') as f:
                new_coco = json.load(f)
            
            self.assertGreater(len(new_coco['images']), 0)
            self.assertGreaterEqual(len(new_coco['annotations']), 0)


# ═══════════════════════════════════════════════════════════════════════
# INTERFAZ DE LÍNEA DE COMANDOS  
# ═══════════════════════════════════════════════════════════════════════

def create_cli_parser():
    """Crea parser para interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        prog='patcher_coco',
        description='Generate patches from COCO dataset for HerdNet training'
    )
    
    parser.add_argument(
        'coco_json',
        type=str,
        nargs='?',  # Hacer opcional
        help='Path to COCO JSON file'
    )
    
    parser.add_argument(
        'images_dir', 
        type=str,
        nargs='?',  # Hacer opcional
        help='Directory containing original images'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        nargs='?',  # Hacer opcional  
        help='Output directory for patches'
    )
    
    parser.add_argument(
        '--patch-size',
        type=int,
        default=512,
        help='Size of square patches (default: 512)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int, 
        default=128,
        help='Overlap between patches in pixels (default: 128)'
    )
    
    parser.add_argument(
        '--min-visibility',
        type=float,
        default=0.1,
        help='Minimum fraction of object area that must be visible (default: 0.1)'
    )
    
    parser.add_argument(
        '--save-all',
        action='store_true',
        help='Save all patches, including those without objects'
    )
    
    parser.add_argument(
        '--output-coco',
        type=str,
        help='Path for output COCO JSON file (default: output_dir/patches_coco.json)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run unit tests'
    )
    
    return parser


def main():
    """Función principal."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Ejecutar tests si se solicita
    if args.test:
        print("🧪 Running unit tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
        return
    
    # Validar argumentos (solo si no es test)
    if not args.coco_json:
        print("❌ COCO JSON file path is required")
        parser.print_help()
        return
        
    if not args.images_dir:
        print("❌ Images directory path is required") 
        parser.print_help()
        return
        
    if not args.output_dir:
        print("❌ Output directory path is required")
        parser.print_help()
        return
    
    if not os.path.exists(args.coco_json):
        print(f"❌ COCO JSON file not found: {args.coco_json}")
        return
    
    if not os.path.exists(args.images_dir):
        print(f"❌ Images directory not found: {args.images_dir}")
        return
    
    # Configurar output COCO path
    if not args.output_coco:
        args.output_coco = os.path.join(args.output_dir, 'patches_coco.json')
    
    print("🔧 COCO Patcher Configuration:")
    print(f"   📁 Input COCO: {args.coco_json}")
    print(f"   📁 Images dir: {args.images_dir}")
    print(f"   📁 Output dir: {args.output_dir}")
    print(f"   🔲 Patch size: {args.patch_size}x{args.patch_size}")
    print(f"   🔄 Overlap: {args.overlap} pixels")
    print(f"   👁️  Min visibility: {args.min_visibility}")
    print(f"   💾 Save all patches: {args.save_all}")
    print(f"   📄 Output COCO: {args.output_coco}")
    print()
    
    # Crear patcher
    patcher = COCOPatcher(
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        save_all_patches=args.save_all
    )
    
    # Procesar dataset
    print("🚀 Starting patch generation...")
    try:
        stats = patcher.process_coco_dataset(
            args.coco_json,
            args.images_dir,
            args.output_dir,
            args.output_coco
        )
        
        print("✅ Patch generation completed!")
        print(stats)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


if __name__ == '__main__':
    main()