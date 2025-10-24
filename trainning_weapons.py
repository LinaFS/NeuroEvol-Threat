import cv2
import os
import csv
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

# ==================== CONFIGURACIÓN ====================
data_dir = 'dataArmas'
output_root = os.path.join('', 'Resultados')
os.makedirs(output_root, exist_ok=True)

annotations_path = os.path.join(output_root, 'anotaciones_Armas.csv')
csv_header = ['image_filename', 'class', 'x_min', 'y_min', 'x_max', 'y_max']
csv_rows = []

# Parámetros de detección AJUSTADOS
MIN_AREA = 200  # Reducido de 500
MAX_AREA_RATIO = 0.95  # Aumentado de 0.90
MIN_CONTOUR_AREA = 300  # Para detección por contornos
EDGE_THRESHOLD1 = 50
EDGE_THRESHOLD2 = 150

# ==================== FUNCIONES DE DETECCIÓN ====================

def obtener_clase_desde_carpeta(carpeta_nombre):
    """Extrae la clase del nombre de la carpeta"""
    nombre_limpio = carpeta_nombre
    nombre_limpio = re.sub(r'_x264.*$', '', nombre_limpio)
    nombre_limpio = re.sub(r'_frames.*$', '', nombre_limpio)
    nombre_limpio = re.sub(r'_video.*$', '', nombre_limpio)
    clase = re.sub(r'\d+$', '', nombre_limpio)
    return clase if clase else carpeta_nombre

def detectar_por_color_y_contraste(image):
    """
    Detección basada en análisis de color y contraste
    Útil para objetos metálicos (armas, cuchillos)
    """
    bboxes = []
    h, w = image.shape[:2]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ecualizar histograma para mejorar contraste
    gray = cv2.equalizeHist(gray)
    
    # Detectar bordes con múltiples umbrales
    edges1 = cv2.Canny(gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
    edges2 = cv2.Canny(gray, 30, 100)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Dilatar para conectar bordes cercanos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > (h * w * MAX_AREA_RATIO):
            continue
        
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # Filtrar por proporción (evitar líneas muy delgadas o muy anchas)
        aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
        if 0.1 < aspect_ratio < 10:
            bboxes.append((x, y, x + w_box, y + h_box))
    
    return bboxes

def detectar_por_segmentacion(image):
    """
    Detección mediante segmentación de color
    Útil para objetos con colores distintivos
    """
    bboxes = []
    h, w = image.shape[:2]
    
    # Convertir a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Crear múltiples máscaras para diferentes rangos de color
    masks = []
    
    # Metales (grises, plateados)
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    masks.append(cv2.inRange(hsv, lower_gray, upper_gray))
    
    # Objetos oscuros (negro, gris oscuro)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    masks.append(cv2.inRange(hsv, lower_dark, upper_dark))
    
    # Objetos brillantes
    lower_bright = np.array([0, 0, 200])
    upper_bright = np.array([180, 30, 255])
    masks.append(cv2.inRange(hsv, lower_bright, upper_bright))
    
    # Combinar todas las máscaras
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > (h * w * MAX_AREA_RATIO):
            continue
        
        x, y, w_box, h_box = cv2.boundingRect(contour)
        bboxes.append((x, y, x + w_box, y + h_box))
    
    return bboxes

def detectar_por_saliencia(image):
    """
    Detección de regiones salientes (prominentes) en la imagen
    """
    bboxes = []
    h, w = image.shape[:2]
    
    # Redimensionar para procesamiento más rápido
    scale = 1.0
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        image_small = cv2.resize(image, None, fx=scale, fy=scale)
    else:
        image_small = image.copy()
    
    # Crear saliency detector
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(image_small)
    
    if not success:
        return bboxes
    
    # Normalizar y binarizar
    saliency_map = (saliency_map * 255).astype(np.uint8)
    _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        h_small, w_small = image_small.shape[:2]
        
        if area < (MIN_CONTOUR_AREA * scale * scale) or area > (h_small * w_small * MAX_AREA_RATIO):
            continue
        
        x, y, w_box, h_box = cv2.boundingRect(contour)
        
        # Escalar de vuelta al tamaño original
        x = int(x / scale)
        y = int(y / scale)
        w_box = int(w_box / scale)
        h_box = int(h_box / scale)
        
        bboxes.append((x, y, x + w_box, y + h_box))
    
    return bboxes

def detectar_por_diferencia_fondo(image):
    """
    Detección asumiendo que el objeto está en primer plano
    """
    bboxes = []
    h, w = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Umbral adaptativo
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > (h * w * MAX_AREA_RATIO):
            continue
        
        x, y, w_box, h_box = cv2.boundingRect(contour)
        bboxes.append((x, y, x + w_box, y + h_box))
    
    return bboxes

def detectar_regiones_interes_basico(image):
    """
    Método básico: dividir imagen en cuadrantes con contenido
    """
    bboxes = []
    h, w = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular varianza en cuadrículas
    grid_size = 4
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < grid_size - 1 else h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < grid_size - 1 else w
            
            cell = gray[y1:y2, x1:x2]
            variance = np.var(cell)
            
            # Si hay suficiente varianza, hay contenido
            if variance > 200:  # Umbral ajustable
                bboxes.append((x1, y1, x2, y2))
    
    return bboxes

def non_max_suppression_simple(boxes, overlap_thresh=0.3):
    """NMS para eliminar cajas superpuestas"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return [tuple(map(int, boxes[i])) for i in pick]

def detectar_objetos_multiple(image):
    """
    Combina TODOS los métodos de detección para máxima cobertura
    """
    h, w = image.shape[:2]
    img_area = h * w
    max_area = img_area * MAX_AREA_RATIO
    
    all_bboxes = []
    
    try:
        # Método 1: Detección por color y contraste
        boxes1 = detectar_por_color_y_contraste(image)
        all_bboxes.extend(boxes1)
    except Exception as e:
        print(f"  Error en método 1: {e}")
    
    try:
        # Método 2: Segmentación por color
        boxes2 = detectar_por_segmentacion(image)
        all_bboxes.extend(boxes2)
    except Exception as e:
        print(f"  Error en método 2: {e}")
    
    try:
        # Método 3: Detección de saliencia
        boxes3 = detectar_por_saliencia(image)
        all_bboxes.extend(boxes3)
    except Exception as e:
        print(f"  Error en método 3: {e}")
    
    try:
        # Método 4: Diferencia de fondo
        boxes4 = detectar_por_diferencia_fondo(image)
        all_bboxes.extend(boxes4)
    except Exception as e:
        print(f"  Error en método 4: {e}")
    
    # Método 5: Regiones de interés básicas (fallback)
    if len(all_bboxes) == 0:
        try:
            boxes5 = detectar_regiones_interes_basico(image)
            all_bboxes.extend(boxes5)
        except Exception as e:
            print(f"  Error en método 5: {e}")
    
    # Filtrar por área
    valid_boxes = []
    for (x1, y1, x2, y2) in all_bboxes:
        area = (x2 - x1) * (y2 - y1)
        if MIN_AREA < area < max_area:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            valid_boxes.append((x1, y1, x2, y2))
    
    # Si no se detectó nada, crear bbox de imagen completa con margen
    if len(valid_boxes) == 0:
        margin = 10
        valid_boxes.append((margin, margin, w - margin, h - margin))
    
    # Eliminar duplicados
    if valid_boxes:
        valid_boxes = non_max_suppression_simple(valid_boxes, 0.4)
    
    return valid_boxes

# ==================== PROCESAMIENTO ====================

print("="*60)
print("GENERADOR DE ANOTACIONES MEJORADO - DETECCIÓN DE OBJETOS")
print("="*60)

if not os.path.exists(data_dir):
    print(f"Error: No existe la carpeta '{data_dir}'")
    exit(1)

video_folders = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]

if not video_folders:
    print(f"No se encontraron subcarpetas en '{data_dir}'")
    exit(1)

print(f"\nCarpetas encontradas: {len(video_folders)}")

# Estadísticas
stats = defaultdict(lambda: {'imagenes': 0, 'detecciones': 0})
imagenes_sin_detecciones = []

# Procesar cada carpeta
for video_folder in tqdm(video_folders, desc="Procesando carpetas"):
    video_path = os.path.join(data_dir, video_folder)
    clase = obtener_clase_desde_carpeta(video_folder)
    
    image_files = sorted([f for f in os.listdir(video_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    stats[clase]['imagenes'] += len(image_files)
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(video_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
        
        # Detectar objetos con múltiples métodos
        bboxes = detectar_objetos_multiple(image)
        
        if len(bboxes) == 0:
            imagenes_sin_detecciones.append(f"{video_folder}/{filename}")
        
        file_ext = os.path.splitext(filename)[1]
        nuevo_nombre = f"{video_folder}_{idx:04d}{file_ext}"
        
        for (x_min, y_min, x_max, y_max) in bboxes:
            csv_rows.append([
                nuevo_nombre,
                clase,
                int(x_min), int(y_min), int(x_max), int(y_max)
            ])
            stats[clase]['detecciones'] += 1

# ==================== GUARDAR RESULTADOS ====================

if csv_rows:
    with open(annotations_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    
    print(f"\nAnotaciones guardadas: {annotations_path}")
    print(f"\nESTADÍSTICAS POR CLASE:")
    print("-" * 70)
    print(f"  {'CLASE':20s} | {'IMÁGENES':>8s} | {'DETECCIONES':>11s} | {'PROM':>6s}")
    print("-" * 70)
    
    total_imgs = 0
    total_dets = 0
    
    for clase in sorted(stats.keys()):
        n_imgs = stats[clase]['imagenes']
        n_dets = stats[clase]['detecciones']
        avg = n_dets / n_imgs if n_imgs > 0 else 0
        
        print(f"  {clase:20s} | {n_imgs:8d} | {n_dets:11d} | {avg:6.2f}")
        total_imgs += n_imgs
        total_dets += n_dets
    
    print("-" * 70)
    avg_total = total_dets/total_imgs if total_imgs > 0 else 0
    print(f"  {'TOTAL':20s} | {total_imgs:8d} | {total_dets:11d} | {avg_total:6.2f}")
    
    if imagenes_sin_detecciones:
<<<<<<< HEAD
        reporte_path = os.path.join(output_root, 'imagenes_sin_detecciones_Armas.txt')
=======
        reporte_path = os.path.join(output_root, 'imagenes_sin_detecciones_armas.txt')
>>>>>>> 884687703343f1b9b7f8d5b84ce82b5d1a76c029
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write(f"Total: {len(imagenes_sin_detecciones)}\n")
            f.write("="*60 + "\n")
            for img in imagenes_sin_detecciones:
                f.write(f"{img}\n")
        
        porcentaje = (len(imagenes_sin_detecciones) / total_imgs) * 100
        print(f"\n⚠  {len(imagenes_sin_detecciones)} imágenes sin detecciones ({porcentaje:.1f}%)")
        print(f"    Reporte: {reporte_path}")
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
else:
    print("\nNo se generaron anotaciones.")