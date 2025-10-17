import cv2
import os
import csv
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

# ==================== CONFIGURACIÓN ====================
data_dir = 'dataSospecha'  # Carpeta principal con subcarpetas de videos
output_root = os.path.join('proyecto', 'resultados')
os.makedirs(output_root, exist_ok=True)

annotations_path = os.path.join(output_root, 'anotaciones.csv')
csv_header = ['image_filename', 'class', 'x_min', 'y_min', 'x_max', 'y_max']
csv_rows = []

# Parámetros de detección
MIN_AREA = 500          # Área mínima del contorno
MAX_AREA_RATIO = 0.85   # Máximo 85% del área de la imagen
MIN_ASPECT_RATIO = 0.2  # Mínima relación ancho/alto
MAX_ASPECT_RATIO = 5.0  # Máxima relación ancho/alto

# ==================== FUNCIONES ====================

def obtener_clase_desde_carpeta(carpeta_nombre):
    """
    Extrae la clase del nombre de la carpeta del video.
    
    Ejemplos:
      - "Abuse001" → "Abuse"
      - "Assault015" → "Assault"
      - "Normal" → "Normal"
      - "RoadAccidents001" → "RoadAccidents"
    """
    # Remover números al final del nombre
    clase = re.sub(r'\d+$', '', carpeta_nombre)
    
    # Si quedó vacío (nombre era solo números), usar el nombre completo
    if not clase:
        clase = carpeta_nombre
    
    return clase

def detectar_objetos_mejorado(image, min_area=500, max_area_ratio=0.85):
    """
    Detección mejorada con validaciones adicionales
    """
    h, w = image.shape[:2]
    img_area = h * w
    max_area = img_area * max_area_ratio
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Método 1: Otsu threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtro de área
        if area < min_area or area > max_area:
            continue
        
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # Validar aspect ratio (no demasiado delgado ni muy ancho)
        aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue
        
        # Evitar bordes de imagen
        margin = 5
        if (x <= margin or y <= margin or 
            x + w_box >= w - margin or y + h_box >= h - margin):
            continue
        
        bboxes.append((x, y, x + w_box, y + h_box))
    
    return bboxes

# ==================== PROCESAMIENTO ====================

print("="*60)
print("GENERADOR DE ANOTACIONES POR VIDEO")
print("="*60)

# Verificar estructura de carpetas
if not os.path.exists(data_dir):
    print(f"❌ Error: No existe la carpeta '{data_dir}'")
    exit(1)

# Obtener todas las carpetas de videos
video_folders = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]

if not video_folders:
    print(f"❌ No se encontraron subcarpetas en '{data_dir}'")
    print("Estructura esperada:")
    print("  dataSospecha/")
    print("    ├── Abuse001/")
    print("    ├── Normal/")
    print("    └── ...")
    exit(1)

print(f"\n📁 Carpetas de videos encontradas: {len(video_folders)}")

# Estadísticas
stats = defaultdict(lambda: {'imagenes': 0, 'detecciones': 0})
imagenes_sin_detecciones = []

# Procesar cada carpeta de video
for video_folder in tqdm(video_folders, desc="Procesando videos"):
    video_path = os.path.join(data_dir, video_folder)
    clase = obtener_clase_desde_carpeta(video_folder)
    
    # Obtener imágenes de esta carpeta
    image_files = [f for f in os.listdir(video_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    stats[clase]['imagenes'] += len(image_files)
    
    # Procesar cada imagen
    for filename in image_files:
        image_path = os.path.join(video_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        # Detectar objetos
        bboxes = detectar_objetos_mejorado(image, MIN_AREA, MAX_AREA_RATIO)
        
        if len(bboxes) == 0:
            # Usar nombre original para el reporte
            imagenes_sin_detecciones.append(f"{video_folder}/{filename}")
        
        # Crear nombre único: nombrecarpeta_idx.ext
        file_ext = os.path.splitext(filename)[1]  # .jpg, .png, etc.
        frame_idx = len([r for r in csv_rows if r[0].startswith(video_folder)])
        nuevo_nombre = f"{video_folder}_{frame_idx:04d}{file_ext}"
        
        # Guardar detecciones con nuevo nombre
        for (x_min, y_min, x_max, y_max) in bboxes:
            csv_rows.append([
                nuevo_nombre,
                clase,
                x_min, y_min, x_max, y_max
            ])
            stats[clase]['detecciones'] += 1

# ==================== GUARDAR RESULTADOS ====================

if csv_rows:
    with open(annotations_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    
    print(f"\n✅ Anotaciones guardadas en: {annotations_path}")
    print(f"\n📊 ESTADÍSTICAS POR CLASE:")
    print("-" * 60)
    
    total_imgs = 0
    total_dets = 0
    
    for clase in sorted(stats.keys()):
        n_imgs = stats[clase]['imagenes']
        n_dets = stats[clase]['detecciones']
        avg = n_dets / n_imgs if n_imgs > 0 else 0
        
        print(f"  {clase:15s} | Imgs: {n_imgs:4d} | Dets: {n_dets:5d} | Avg: {avg:.2f}")
        total_imgs += n_imgs
        total_dets += n_dets
    
    print("-" * 60)
    print(f"  {'TOTAL':15s} | Imgs: {total_imgs:4d} | Dets: {total_dets:5d} | Avg: {total_dets/total_imgs:.2f}")
    
    # Guardar reporte de imágenes sin detecciones
    if imagenes_sin_detecciones:
        reporte_path = os.path.join(output_root, 'imagenes_sin_detecciones.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write(f"Total: {len(imagenes_sin_detecciones)}\n")
            f.write("="*60 + "\n")
            for img in imagenes_sin_detecciones:
                f.write(f"{img}\n")
        print(f"\n⚠️  {len(imagenes_sin_detecciones)} imágenes sin detecciones")
        print(f"    Ver: {reporte_path}")
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
else:
    print("\n❌ No se generaron anotaciones. Verifica los parámetros de detección.")