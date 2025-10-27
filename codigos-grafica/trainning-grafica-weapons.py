import cv2
import os
import csv
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict
import gc

# ==================== CONFIGURACIÓN ====================
data_dir = 'dataArmas'
output_root = os.path.join('', 'Resultados')
os.makedirs(output_root, exist_ok=True)

annotations_path = os.path.join(output_root, 'anotaciones_Armas.csv')
csv_header = ['image_filename', 'class', 'x_min', 'y_min', 'x_max', 'y_max']
csv_rows = []

# Parámetros de detección AJUSTADOS
MIN_AREA = 200
MAX_AREA_RATIO = 0.95
MIN_CONTOUR_AREA = 300
EDGE_THRESHOLD1 = 50
EDGE_THRESHOLD2 = 150

# ==================== CONFIGURACIÓN GPU ====================
USE_GPU = True  # Cambiar a False para usar solo CPU
USE_CUDA = False  # CUDA para operaciones avanzadas (requiere opencv-contrib-python compilado con CUDA)

def verificar_opencl():
    """Verifica si OpenCL está disponible en el sistema"""
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print("✅ OpenCL disponible y activado")
            
            if cv2.ocl.useOpenCL():
                device = cv2.ocl.Device.getDefault()
                print(f"   Dispositivo: {device.name()}")
                print(f"   Tipo: {device.type()}")
                return True
        else:
            print("⚠️  OpenCL no disponible")
            return False
    except Exception as e:
        print(f"⚠️  Error verificando OpenCL: {e}")
        return False

def verificar_cuda():
    """Verifica si CUDA está disponible"""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✅ CUDA disponible - {cv2.cuda.getCudaEnabledDeviceCount()} dispositivo(s)")
            return True
        else:
            print("⚠️  CUDA no disponible")
            return False
    except:
        print("⚠️  CUDA no disponible en esta versión de OpenCV")
        return False

# Inicializar GPU
print("🔧 Inicializando aceleración GPU...")
opencl_disponible = verificar_opencl() if USE_GPU else False
cuda_disponible = verificar_cuda() if USE_CUDA else False

# ==================== FUNCIONES DE DETECCIÓN ====================

def obtener_clase_desde_carpeta(carpeta_nombre):
    """Extrae la clase del nombre de la carpeta"""
    nombre_limpio = carpeta_nombre
    nombre_limpio = re.sub(r'_x264.*$', '', nombre_limpio)
    nombre_limpio = re.sub(r'_frames.*$', '', nombre_limpio)
    nombre_limpio = re.sub(r'_video.*$', '', nombre_limpio)
    clase = re.sub(r'\d+$', '', nombre_limpio)
    return clase if clase else carpeta_nombre

def detectar_por_color_y_contraste(image, usar_gpu=False):
    """
    Detección basada en análisis de color y contraste (GPU optimizado)
    Útil para objetos metálicos (armas, cuchillos)
    """
    bboxes = []
    h, w = image.shape[:2]
    
    try:
        # Usar UMat para procesamiento GPU si está disponible
        if usar_gpu and opencl_disponible:
            img_gpu = cv2.UMat(image)
            gray = cv2.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detectar bordes
            edges1 = cv2.Canny(gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
            edges2 = cv2.Canny(gray, 30, 100)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Dilatar
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Convertir de vuelta a numpy para findContours
            edges_cpu = edges.get()
        else:
            # Procesamiento CPU
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            edges1 = cv2.Canny(gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
            edges2 = cv2.Canny(gray, 30, 100)
            edges = cv2.bitwise_or(edges1, edges2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges_cpu = edges
        
        # Encontrar contornos (siempre en CPU)
        contours, _ = cv2.findContours(edges_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA or area > (h * w * MAX_AREA_RATIO):
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
            if 0.1 < aspect_ratio < 10:
                bboxes.append((x, y, x + w_box, y + h_box))
    except Exception as e:
        # Si falla GPU, reintentar en CPU
        if usar_gpu:
            return detectar_por_color_y_contraste(image, usar_gpu=False)
    
    return bboxes

def detectar_por_segmentacion(image, usar_gpu=False):
    """
    Detección mediante segmentación de color (GPU optimizado)
    """
    bboxes = []
    h, w = image.shape[:2]
    
    try:
        if usar_gpu and opencl_disponible:
            img_gpu = cv2.UMat(image)
            hsv = cv2.cvtColor(img_gpu, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Crear máscaras
        masks = []
        
        # Metales (grises, plateados)
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        masks.append(cv2.inRange(hsv, lower_gray, upper_gray))
        
        # Objetos oscuros
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        masks.append(cv2.inRange(hsv, lower_dark, upper_dark))
        
        # Objetos brillantes
        lower_bright = np.array([0, 0, 200])
        upper_bright = np.array([180, 30, 255])
        masks.append(cv2.inRange(hsv, lower_bright, upper_bright))
        
        # Combinar máscaras
        if usar_gpu and opencl_disponible:
            combined_mask = cv2.UMat(np.zeros_like(masks[0].get() if hasattr(masks[0], 'get') else masks[0]))
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        else:
            combined_mask = np.zeros_like(masks[0])
            for mask in masks:
                mask_cpu = mask.get() if hasattr(mask, 'get') else mask
                combined_mask = cv2.bitwise_or(combined_mask, mask_cpu)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Convertir a CPU para findContours
        mask_cpu = combined_mask.get() if hasattr(combined_mask, 'get') else combined_mask
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA or area > (h * w * MAX_AREA_RATIO):
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            bboxes.append((x, y, x + w_box, y + h_box))
    except Exception as e:
        if usar_gpu:
            return detectar_por_segmentacion(image, usar_gpu=False)
    
    return bboxes

def detectar_por_saliencia(image):
    """
    Detección de regiones salientes (siempre en CPU - no hay versión GPU)
    """
    bboxes = []
    h, w = image.shape[:2]
    
    try:
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
            
            # Escalar de vuelta
            x = int(x / scale)
            y = int(y / scale)
            w_box = int(w_box / scale)
            h_box = int(h_box / scale)
            
            bboxes.append((x, y, x + w_box, y + h_box))
    except:
        pass
    
    return bboxes

def detectar_por_diferencia_fondo(image, usar_gpu=False):
    """
    Detección por diferencia de fondo (GPU optimizado)
    """
    bboxes = []
    h, w = image.shape[:2]
    
    try:
        if usar_gpu and opencl_disponible:
            img_gpu = cv2.UMat(image)
            gray = cv2.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            thresh_cpu = thresh.get()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh_cpu = thresh
        
        contours, _ = cv2.findContours(thresh_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA or area > (h * w * MAX_AREA_RATIO):
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            bboxes.append((x, y, x + w_box, y + h_box))
    except Exception as e:
        if usar_gpu:
            return detectar_por_diferencia_fondo(image, usar_gpu=False)
    
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
            
            if variance > 200:
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
    Combina TODOS los métodos de detección con aceleración GPU
    """
    h, w = image.shape[:2]
    img_area = h * w
    max_area = img_area * MAX_AREA_RATIO
    
    all_bboxes = []
    usar_gpu = USE_GPU and opencl_disponible
    
    # Método 1: Color y contraste (GPU)
    try:
        boxes1 = detectar_por_color_y_contraste(image, usar_gpu)
        all_bboxes.extend(boxes1)
    except:
        pass
    
    # Método 2: Segmentación (GPU)
    try:
        boxes2 = detectar_por_segmentacion(image, usar_gpu)
        all_bboxes.extend(boxes2)
    except:
        pass
    
    # Método 3: Saliencia (CPU - no hay GPU)
    try:
        boxes3 = detectar_por_saliencia(image)
        all_bboxes.extend(boxes3)
    except:
        pass
    
    # Método 4: Diferencia de fondo (GPU)
    try:
        boxes4 = detectar_por_diferencia_fondo(image, usar_gpu)
        all_bboxes.extend(boxes4)
    except:
        pass
    
    # Método 5: Fallback básico
    if len(all_bboxes) == 0:
        try:
            boxes5 = detectar_regiones_interes_basico(image)
            all_bboxes.extend(boxes5)
        except:
            pass
    
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
    
    # Si no se detectó nada, bbox completa
    if len(valid_boxes) == 0:
        margin = 10
        valid_boxes.append((margin, margin, w - margin, h - margin))
    
    # NMS
    if valid_boxes:
        valid_boxes = non_max_suppression_simple(valid_boxes, 0.4)
    
    return valid_boxes

# ==================== PROCESAMIENTO ====================

print("="*60)
print("DETECTOR DE ARMAS - GPU ACELERADO")
print("="*60)

if not os.path.exists(data_dir):
    print(f"❌ Error: No existe la carpeta '{data_dir}'")
    exit(1)

video_folders = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]

if not video_folders:
    print(f"❌ No se encontraron subcarpetas en '{data_dir}'")
    exit(1)

print(f"\n📁 Carpetas encontradas: {len(video_folders)}")

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
        
        # Detectar objetos
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
        
        # Liberar memoria
        del image
        if idx % 100 == 0:
            gc.collect()

# ==================== GUARDAR RESULTADOS ====================

if csv_rows:
    with open(annotations_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    
    print(f"\n✅ Anotaciones guardadas: {annotations_path}")
    print(f"\n📊 ESTADÍSTICAS POR CLASE:")
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
        reporte_path = os.path.join(output_root, 'imagenes_sin_detecciones_armas.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write(f"Total: {len(imagenes_sin_detecciones)}\n")
            f.write("="*60 + "\n")
            for img in imagenes_sin_detecciones:
                f.write(f"{img}\n")
        
        porcentaje = (len(imagenes_sin_detecciones) / total_imgs) * 100
        print(f"\n⚠️  {len(imagenes_sin_detecciones)} imágenes sin detecciones ({porcentaje:.1f}%)")
        print(f"    Reporte: {reporte_path}")
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
else:
    print("\n❌ No se generaron anotaciones.")