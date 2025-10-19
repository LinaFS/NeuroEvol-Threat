import cv2
import os
import csv
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

# ==================== CONFIGURACIÓN ====================
data_dir = 'dataSospecha'
output_root = os.path.join('', 'resultados')
os.makedirs(output_root, exist_ok=True)

annotations_path = os.path.join(output_root, 'anotaciones.csv')
csv_header = ['image_filename', 'class', 'x_min', 'y_min', 'x_max', 'y_max']
csv_rows = []

# Parámetros de detección
MIN_CONFIDENCE = 0.3
MIN_AREA = 500
MAX_AREA_RATIO = 0.90

# ==================== CONFIGURACIÓN GPU ====================
USE_GPU = True  # Cambiar a False para usar solo CPU

def configurar_gpu():
    """Detecta y configura el backend de GPU disponible"""
    backends_gpu = [
        (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA (NVIDIA)"),
        (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16, "CUDA FP16 (NVIDIA)"),
        (cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv2.dnn.DNN_TARGET_GPU, "OpenVINO (Intel)"),
        (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL, "OpenCL (Universal)"),
    ]
    
    if not USE_GPU:
        print("🔧 Modo CPU seleccionado")
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
    
    print("🔍 Detectando GPU disponible...")
    
    # Probar cada backend
    for backend, target, nombre in backends_gpu:
        try:
            # Crear una red temporal para probar
            net_test = cv2.dnn.readNetFromCaffe(
                cv2.samples.findFile('face_detector/deploy.prototxt'),
                cv2.samples.findFile('face_detector/res10_300x300_ssd_iter_140000.caffemodel')
            )
            net_test.setPreferableBackend(backend)
            net_test.setPreferableTarget(target)
            
            # Probar una inferencia pequeña
            blob = cv2.dnn.blobFromImage(np.zeros((300, 300, 3), dtype=np.uint8))
            net_test.setInput(blob)
            net_test.forward()
            
            print(f"✅ GPU detectada: {nombre}")
            return backend, target
        except:
            continue
    
    print("⚠️  No se detectó GPU compatible, usando CPU")
    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU

# ==================== INICIALIZAR DETECTORES ====================

print("🔧 Inicializando detectores...")

# Configurar GPU
gpu_backend, gpu_target = configurar_gpu()

# Opción 1: YOLO con GPU
USE_YOLO = False
yolo_net = None
yolo_output_layers = None
yolo_classes = None

if USE_YOLO and os.path.exists('yolov3.weights'):
    yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    yolo_net.setPreferableBackend(gpu_backend)
    yolo_net.setPreferableTarget(gpu_target)
    layer_names = yolo_net.getLayerNames()
    yolo_output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    with open('coco.names', 'r') as f:
        yolo_classes = [line.strip() for line in f.readlines()]
    print("✅ YOLO cargado con aceleración GPU")

# Opción 2: HOG Detector (personas) - se mantiene en CPU
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("✅ HOG Detector cargado (personas)")

# Opción 3: Haar Cascade - aceleración mediante UMat
USE_UMAT = True and USE_GPU  # Usar UMat si GPU está disponible
cascade_fullbody = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
cascade_upperbody = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ Haar Cascades cargados" + (" con UMat (GPU)" if USE_UMAT else ""))

# ==================== FUNCIONES ====================

def obtener_clase_desde_carpeta(carpeta_nombre):
    """
    Extrae la clase del nombre de la carpeta.
    """
    nombre_limpio = carpeta_nombre
    nombre_limpio = re.sub(r'_x264.*$', '', nombre_limpio)
    nombre_limpio = re.sub(r'_frames.*$', '', nombre_limpio)
    nombre_limpio = re.sub(r'_video.*$', '', nombre_limpio)
    
    clase = re.sub(r'\d+$', '', nombre_limpio)
    
    if not clase:
        clase = carpeta_nombre
    
    return clase

def detectar_con_yolo(image):
    """Detección con YOLO (GPU acelerado)"""
    if yolo_net is None:
        return []
    
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(yolo_output_layers)
    
    bboxes = []
    confidences = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > MIN_CONFIDENCE and class_id == 0:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                w_box = int(detection[2] * w)
                h_box = int(detection[3] * h)
                
                x = int(center_x - w_box / 2)
                y = int(center_y - h_box / 2)
                
                bboxes.append((x, y, x + w_box, y + h_box))
                confidences.append(float(confidence))
    
    if bboxes:
        indices = cv2.dnn.NMSBoxes(
            [(b[0], b[1], b[2]-b[0], b[3]-b[1]) for b in bboxes],
            confidences, MIN_CONFIDENCE, 0.4
        )
        if len(indices) > 0:
            bboxes = [bboxes[i] for i in indices.flatten()]
    
    return bboxes

def detectar_con_hog(image):
    """Detección de personas con HOG"""
    try:
        scale = 1.0
        if image.shape[0] > 600:
            scale = 600.0 / image.shape[0]
            image_resized = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            image_resized = image
        
        bboxes, weights = hog.detectMultiScale(
            image_resized, 
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0
        )
        
        scaled_bboxes = []
        for (x, y, w, h) in bboxes:
            x_scaled = int(x / scale)
            y_scaled = int(y / scale)
            w_scaled = int(w / scale)
            h_scaled = int(h / scale)
            scaled_bboxes.append((x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled))
        
        return scaled_bboxes
    except Exception as e:
        return []

def detectar_con_haarcascade(image):
    """Detección con Haar Cascades (GPU acelerado con UMat)"""
    # Convertir a UMat para procesamiento GPU si está disponible
    if USE_UMAT:
        img_umat = cv2.UMat(image)
        gray = cv2.cvtColor(img_umat, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    
    all_detections = []
    
    # Detectar cuerpo completo
    bodies = cascade_fullbody.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 90))
    for (x, y, w, h) in bodies:
        all_detections.append((x, y, x+w, y+h))
    
    # Detectar torso
    upper_bodies = cascade_upperbody.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 60))
    for (x, y, w, h) in upper_bodies:
        all_detections.append((x, y, x+w, y+h))
    
    # Detectar rostros
    faces = cascade_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        x_new = max(0, x - w//2)
        y_new = max(0, y - h//2)
        w_new = min(image.shape[1] - x_new, w * 2)
        h_new = min(image.shape[0] - y_new, h * 3)
        all_detections.append((x_new, y_new, x_new + w_new, y_new + h_new))
    
    return all_detections

def non_max_suppression_simple(boxes, overlap_thresh=0.3):
    """NMS simple para eliminar cajas superpuestas"""
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

def detectar_objetos_inteligente(image):
    """
    Combina múltiples detectores para máxima cobertura
    """
    h, w = image.shape[:2]
    img_area = h * w
    max_area = img_area * MAX_AREA_RATIO
    
    all_bboxes = []
    
    # 1. Intentar YOLO si está disponible (GPU)
    if USE_YOLO:
        yolo_boxes = detectar_con_yolo(image)
        all_bboxes.extend(yolo_boxes)
    
    # 2. HOG para personas
    hog_boxes = detectar_con_hog(image)
    all_bboxes.extend(hog_boxes)
    
    # 3. Haar Cascades (GPU con UMat)
    haar_boxes = detectar_con_haarcascade(image)
    all_bboxes.extend(haar_boxes)
    
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
    
    # Eliminar duplicados
    if valid_boxes:
        valid_boxes = non_max_suppression_simple(valid_boxes, 0.3)
    
    return valid_boxes

# ==================== PROCESAMIENTO ====================

print("="*60)
print("GENERADOR DE ANOTACIONES - DETECCIÓN GPU ACELERADA")
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
print("\n📋 Preview de clases:")
for vf in sorted(video_folders)[:10]:
    clase = obtener_clase_desde_carpeta(vf)
    print(f"  {vf:30s} → {clase}")
if len(video_folders) > 10:
    print(f"  ... y {len(video_folders) - 10} más")

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
        
        # Detectar objetos/personas
        bboxes = detectar_objetos_inteligente(image)
        
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
        reporte_path = os.path.join(output_root, 'imagenes_sin_detecciones.txt')
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write(f"Total: {len(imagenes_sin_detecciones)}\n")
            f.write("="*60 + "\n")
            for img in imagenes_sin_detecciones:
                f.write(f"{img}\n")
        
        porcentaje = (len(imagenes_sin_detecciones) / total_imgs) * 100
        print(f"\n⚠  {len(imagenes_sin_detecciones)} imágenes sin detecciones ({porcentaje:.1f}%)")
        print(f"    Reporte: {reporte_path}")
        
        if porcentaje > 50:
            print("\n💡 SUGERENCIAS:")
            print(f"   - Reducir MIN_AREA (actualmente {MIN_AREA})")
            print(f"   - Reducir MIN_CONFIDENCE (actualmente {MIN_CONFIDENCE})")
            print("   - Considerar usar YOLO (más preciso)")
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
else:
    print("\n❌ No se generaron anotaciones.")