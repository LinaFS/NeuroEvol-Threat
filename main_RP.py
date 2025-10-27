"""
main_RP.py - VERSIÓN 4 (Profesional)
Correcciones:
- Reemplazado SimpleTracker con el tracker profesional ByteTrack (model.track()).
- Aumentados los umbrales de confianza para eliminar falsos positivos.
- Añadida lógica para ASOCIAR armas/comportamientos a los IDs de las personas.
- Mantiene la optimización (NMS y ejecución jerárquica).
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque, defaultdict

# ============================================
# ¡NUEVO! IMPORTAR TU CONFIGURACIÓN DE CLASES
# ============================================
# (Asegúrate que classes_config.py esté en la misma carpeta)
try:
    import classes_config as cfg
except ImportError:
    print("❌ ERROR: No se pudo encontrar el archivo 'classes_config.py'")
    print("Asegúrate de que esté en la misma carpeta que main_RP.py")
    exit()

# ============================================
# CONFIGURACIÓN
# ============================================
MODELO_GENERAL = 'yolov8n.pt'
# ¡Asegúrate que esta sea la ruta con 'e' (ModeloSopecha...) que vimos!
MODELO_SOSPECHOSO = r'C:\Users\admi\Downloads\NeuroEvol-Threat-master\ModeloSopechaOptimizado\best_model_ga_optimized\weights\best.pt'
MODELO_ARMAS = r'C:\Users\admi\Downloads\NeuroEvol-Threat-master\ModeloArmasOptimizado\best_model_ga_optimized\weights\best.pt'

# --- ¡MODIFICADO! UMBRALES DE CONFIANZA MÁS ALTOS ---
CONFIDENCE_GENERAL = 0.4    # Confianza mínima para detectar una persona
CONFIDENCE_ARMAS = 0.60     # ¡Subido de 0.25 a 0.60! Elimina falsos positivos
CONFIDENCE_SOSPECHOSO = 0.65 # ¡Subido de 0.4 a 0.65!

# Umbrales de NMS (para limpiar detecciones duplicadas)
NMS_IOU_ARMAS = 0.3
NMS_IOU_SOSPECHOSO = 0.4

# Parámetros de asociación
IOU_ASSOCIATION_THRESHOLD = 0.1 # Mínimo solapamiento para asociar un arma a una persona

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def bbox_overlap(bbox1, bbox2):
    """Calcular IoU (Intersection over Union) entre dos bboxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def apply_nms(detections, iou_threshold):
    """Aplica Non-Max Suppression (NMS) a una lista de detecciones."""
    if len(detections) == 0:
        return []

    boxes = np.array([d[:4] for d in detections])
    scores = np.array([d[4] for d in detections])
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        union = (areas[i] + areas[order[1:]] - intersection) + 1e-6
        iou = intersection / union
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    final_detections = [detections[i] for i in keep]
    return final_detections

# ============================================
# CLASE: SIMPLE TRACKER (ELIMINADA)
# (Ya no se necesita, usamos el tracker de YOLO)
# ============================================

# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main():
    print("🔧 Cargando modelos...")
    modelo_general = YOLO(MODELO_GENERAL)
    modelo_armas = YOLO(MODELO_ARMAS)
    modelo_sospechoso = YOLO(MODELO_SOSPECHOSO)
    print("✅ Modelos cargados")
    
    # --- Tracker eliminado, ya no se inicializa ---
    
    cap = cv2.VideoCapture(0) # 0 para webcam
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    frame_count = 0
    print("\n🚀 Sistema iniciado. Presiona 'ESC' para salir.")
    cfg.print_classes_summary()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- PASO 1: OPTIMIZAR LA ENTRADA ---
            try:
                frame_procesado = cv2.resize(frame, (640, 480))
            except cv2.error:
                continue
            frame_display = frame_procesado.copy()
            
            # --- FASE 1: TRACKING DE PERSONAS (NUEVA LÓGICA) ---
            # Usamos model.track() para detectar y rastrear personas (clase 0)
            # 'persist=True' le dice al tracker que recuerde los IDs entre frames
            results_general = modelo_general.track(
                frame_procesado, 
                persist=True, 
                classes=0,                  # Solo rastrear clase 0 (personas)
                conf=CONFIDENCE_GENERAL,    # Usar nuestro umbral de confianza
                verbose=False
            )[0]

            # Obtener los tracks de personas (si existen)
            tracked_persons = []
            if results_general.boxes.id is not None:
                for box in results_general.boxes:
                    tracked_persons.append({
                        'id': int(box.id[0]),
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'conf': float(box.conf[0])
                    })
            
            # --- FASE 2: DETECCIÓN DE ARMAS Y COMPORTAMIENTOS ---
            detecciones_armas = []
            detecciones_sospechoso = []
            
            # Solo ejecutamos los modelos pesados si hay personas en escena
            if len(tracked_persons) > 0:
                # --- Modelo Armas ---
                resultados_armas = modelo_armas(frame_procesado, verbose=False)[0]
                detecciones_armas_bruto = [
                    r for r in resultados_armas.boxes.data.cpu().numpy() 
                    if r[4] > CONFIDENCE_ARMAS
                ]
                detecciones_armas = apply_nms(detecciones_armas_bruto, iou_threshold=NMS_IOU_ARMAS)
                
                # --- Modelo Comportamientos ---
                resultados_sospechoso = modelo_sospechoso(frame_procesado, verbose=False)[0]
                detecciones_sospechoso_bruto = [
                    r for r in resultados_sospechoso.boxes.data.cpu().numpy()
                    if r[4] > CONFIDENCE_SOSPECHOSO
                ]
                detecciones_sospechoso = apply_nms(detecciones_sospechoso_bruto, iou_threshold=NMS_IOU_SOSPECHOSO)

            # --- FASE 3: ASOCIACIÓN Y VISUALIZACIÓN ---
            
            # Sets para rastrear detecciones que ya han sido asociadas a una persona
            used_armas_idx = set()
            used_sospechoso_idx = set()

            # 1. Dibujar Personas y Alertas Asociadas
            for person in tracked_persons:
                person_id = person['id']
                person_bbox = person['bbox']
                
                # Valores por defecto (persona normal)
                display_label = f"Persona ID:{person_id}"
                display_color = (0, 255, 0) # Verde
                display_riesgo = 0

                # Buscar armas asociadas a esta persona
                for i, det_arma in enumerate(detecciones_armas):
                    if i in used_armas_idx: continue
                    
                    arma_bbox = det_arma[:4]
                    if bbox_overlap(person_bbox, arma_bbox) > IOU_ASSOCIATION_THRESHOLD:
                        class_id = int(det_arma[5])
                        nombre_arma = cfg.get_class_name(class_id, is_weapon=True)
                        riesgo = cfg.get_risk_level(class_id, is_weapon=True)
                        
                        display_label = f"ID:{person_id} - {nombre_arma.upper()}"
                        display_color = cfg.get_class_color(nombre_arma, is_weapon=True)
                        display_riesgo = riesgo
                        used_armas_idx.add(i)
                        break # Asociar solo la primera/mejor arma
                
                # Si no hay arma, buscar comportamiento sospechoso asociado
                if display_riesgo == 0:
                    for j, det_comp in enumerate(detecciones_sospechoso):
                        if j in used_sospechoso_idx: continue
                        
                        comp_bbox = det_comp[:4]
                        if bbox_overlap(person_bbox, comp_bbox) > IOU_ASSOCIATION_THRESHOLD:
                            class_id = int(det_comp[5])
                            nombre_clase = cfg.get_class_name(class_id, is_weapon=False)
                            riesgo = cfg.get_risk_level(class_id, is_weapon=False)
                            
                            display_label = f"ID:{person_id} - R{riesgo}: {nombre_clase}"
                            display_color = cfg.get_class_color(nombre_clase, is_weapon=False)
                            display_riesgo = riesgo
                            used_sospechoso_idx.add(j)
                            break # Asociar solo el primer comportamiento

                # Dibujar el bounding box de la persona con la etiqueta y color correctos
                x1, y1, x2, y2 = person_bbox.astype(int)
                grosor = 4 if display_riesgo > 0 else 2
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), display_color, grosor)
                cv2.putText(frame_display, display_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)

            # 2. Dibujar Alertas "Huérfanas" (no asociadas a ninguna persona)
            for i, det_arma in enumerate(detecciones_armas):
                if i not in used_armas_idx: # Si no fue usada, dibujarla
                    x1, y1, x2, y2, conf, cls = det_arma
                    class_id = int(cls)
                    nombre_arma = cfg.get_class_name(class_id, is_weapon=True)
                    color = cfg.get_class_color(nombre_arma, is_weapon=True)
                    riesgo = cfg.get_risk_level(class_id, is_weapon=True)
                    label = f"R{riesgo}: {nombre_arma.upper()} ({conf:.2f})"
                    
                    cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(frame_display, label, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)

            for j, det_comp in enumerate(detecciones_sospechoso):
                if j not in used_sospechoso_idx: # Si no fue usada, dibujarla
                    x1, y1, x2, y2, conf, cls = det_comp
                    class_id = int(cls)
                    nombre_clase = cfg.get_class_name(class_id, is_weapon=False)
                    color = cfg.get_class_color(nombre_clase, is_weapon=False)
                    riesgo = cfg.get_risk_level(class_id, is_weapon=False)
                    label = f"R{riesgo}: {nombre_clase} ({conf:.2f})"

                    cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(frame_display, label, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # --- ¡MODIFICADO! Info general ---
            info_text = f"Personas: {len(tracked_persons)} | Armas: {len(detecciones_armas)} | Comportamientos: {len(detecciones_sospechoso)} | Frame: {frame_count}"
            cv2.putText(frame_display, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('NeuroEvol-Threat - Reconocimiento de Patrones', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            
            frame_count += 1
    
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("━" * 60)
        print("✅ Sistema finalizado")


if __name__ == "__main__":
    import multiprocessing
    # Esta línea es importante para que funcione si lo compilas en un .exe
    multiprocessing.freeze_support() 
    main()