"""
main.py - VERSIÓN CORREGIDA
Correcciones:
- *** RE-AJUSTE DE SENSIBILIDAD (BALANCEO) ***
- 1. Aumentada sensibilidad de MODELO_SOSPECHOSO (0.5) para que detecte acciones.
- 2. Reducida sensibilidad de BehaviorAnalyzer (movimiento) para evitar "RUNNING" falsos.
"""

print("--- SCRIPT CARGADO POR PYTHON ---") # <-- Diagnóstico

from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque, defaultdict
from pathlib import Path
import argparse

# Importar configuración de clases
from classes_config import (
    BEHAVIOR_CLASSES, 
    WEAPON_CLASSES,
    CLASS_RISK_LEVELS,
    WEAPON_RISK_LEVELS,
    get_risk_level,
    get_class_name,
    get_class_color,
    get_temporal_category,
    should_analyze_temporally
)

# ============================================
# ¡NUEVO! IMPORTAR TU CONFIGURACIÓN DE CLASES
# ============================================
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
MODELO_SOSPECHOSO = r'C:\Users\admi\Downloads\NeuroEvol-Threat-master\ModeloSopechaOptimizado\best_model_ga_optimized\weights\best.pt'
MODELO_ARMAS = r'C:\Users\admi\Downloads\NeuroEvol-Threat-master\ModeloArmasOptimizado\best_model_ga_optimized\weights\best.pt'


# Parámetros de tracking
MAX_DISAPPEARED = 30
DISTANCE_THRESHOLD = 50

# Parámetros de análisis temporal (AJUSTADOS)
WINDOW_SIZE = 15
LOITERING_TIME = 10  # <-- ### NUEVA CORRECCIÓN ### Aumentado a 10s
VELOCITY_THRESHOLD = 3 # (Esta variable no se usa, el valor real está en la clase)

# --- RE-AJUSTE DE UMBRALES ---
CONFIDENCE_GENERAL = 0.3
CONFIDENCE_SOSPECHOSO = 0.5 # <-- ### NUEVA CORRECCIÓN ### Bajado a 0.5 (punto medio)
CONFIDENCE_ARMAS = 0.70   # <-- Mantenemos este alto

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


def bbox_distance(bbox1, bbox2):
    """Calcular distancia entre centroides de dos bboxes"""
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


# ============================================
# CLASE: SIMPLE TRACKER
# ============================================
class SimpleTracker:
    """Tracker simple basado en distancia euclidiana"""
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.trajectories = defaultdict(lambda: deque(maxlen=90))
        self.max_disappeared = max_disappeared
        
    def register(self, centroid, bbox):
        """Registrar nuevo objeto"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trajectories[self.next_id].append({
            'centroid': centroid,
            'bbox': bbox,
            'timestamp': time.time()
        })
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id):
        """Eliminar objeto perdido"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """
        Actualizar tracker con nuevas detecciones
        detections: List[(x1, y1, x2, y2, confidence, class)]
        Returns: Dict[id: (x1, y1, x2, y2, class)]
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # Calcular centroides
        input_centroids = []
        input_bboxes = []
        input_classes = []
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy))
            input_bboxes.append((x1, y1, x2, y2))
            input_classes.append(int(cls))
        
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calcular distancias
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i, j] = np.linalg.norm(
                        np.array(obj_centroid) - np.array(input_centroid)
                    )
            
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if distances[row, col] > DISTANCE_THRESHOLD:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append({
                    'centroid': input_centroids[col],
                    'bbox': input_bboxes[col],
                    'timestamp': time.time()
                })
                
                used_rows.add(row)
                used_cols.add(col)
            
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])
        
        active_objects = {}
        for object_id in self.objects.keys():
            if len(self.trajectories[object_id]) > 0:
                last_point = self.trajectories[object_id][-1]
                active_objects[object_id] = last_point['bbox']
        
        return active_objects


# ============================================
# CLASE: ANALIZADOR DE COMPORTAMIENTO
# ============================================
class BehaviorAnalyzer:
    """Analiza trayectorias para detectar comportamientos sospechosos"""
    def __init__(self):
        self.alert_cooldown = {}
        self.cooldown_time = 5
    
    def analyze_trajectory(self, trajectory, track_id, weapon_detected=False):
        """
        Analizar trayectoria y determinar comportamiento
        Returns: (behavior, alert_level, features)
        """
        # --- ### NUEVA CORRECCIÓN ###: Requerir más frames para ser más estable
        if len(trajectory) < 5: 
            return 'normal', 0, {}
        
        features = self._extract_features(trajectory)
        
        behavior = 'normal'
        alert_level = 0
        
        # 1. PORTACIÓN DE ARMA (Prioridad máxima)
        if weapon_detected:
            behavior = 'weapon_carry'
            alert_level = 3
        
        # --- ### NUEVA CORRECCIÓN ###: Hacer LOITERING menos sensible
        elif features['dwelling_time'] > LOITERING_TIME and features['velocity_mean'] < 0.5: # <-- Más estricto (0.5)
            behavior = 'loitering'
            alert_level = 2
        
        # --- ### NUEVA CORRECCIÓN ###: Hacer ERRATIC menos sensible
        elif features['direction_changes'] > 8 and features['velocity_std'] > 3.0: # <-- Más estricto (8 cambios)
            behavior = 'erratic_movement'
            alert_level = 2
        
        # --- ### NUEVA CORRECCIÓN ###: Hacer RUNNING menos sensible
        elif features['velocity_mean'] > 12.0: # <-- Más estricto (12.0)
            behavior = 'running'
            alert_level = 1
        
        # Verificar cooldown
        if alert_level > 0:
            current_time = time.time()
            if track_id in self.alert_cooldown:
                if current_time - self.alert_cooldown[track_id] < self.cooldown_time:
                    alert_level = 0
            self.alert_cooldown[track_id] = current_time
        
        return behavior, alert_level, features
    
    def _extract_features(self, trajectory):
        """Extraer características de la trayectoria"""
        features = {}
        
        # Velocidades
        velocities = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]['centroid']
            curr = trajectory[i]['centroid']
            dt = trajectory[i]['timestamp'] - trajectory[i-1]['timestamp']
            
            if dt > 0:
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        features['velocity_mean'] = np.mean(velocities) if velocities else 0
        features['velocity_std'] = np.std(velocities) if velocities else 0
        features['velocity_max'] = np.max(velocities) if velocities else 0
        
        # Tiempo de permanencia
        total_time = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
        features['dwelling_time'] = total_time
        
        # Cambios de dirección
        direction_changes = 0
        for i in range(2, len(trajectory)):
            v1 = np.array(trajectory[i-1]['centroid']) - np.array(trajectory[i-2]['centroid'])
            v2 = np.array(trajectory[i]['centroid']) - np.array(trajectory[i-1]['centroid'])
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                if angle > 45:
                    direction_changes += 1
        
        features['direction_changes'] = direction_changes
        
        # Distancia total
        total_distance = 0
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]['centroid']
            curr = trajectory[i]['centroid']
            total_distance += np.linalg.norm(np.array(curr) - np.array(prev))
        
        features['distance_traveled'] = total_distance
        
        return features


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main(video_source):
    # Cargar modelos
    print("🔧 Cargando modelos...")
    modelo_general = YOLO(MODELO_GENERAL)
    modelo_armas = YOLO(MODELO_ARMAS)
    
    try:
        modelo_sospechoso = YOLO(MODELO_SOSPECHOSO)
    except Exception as e:
        print(f"❌ ERROR: No se pudo cargar el modelo 'MODELO_SOSPECHOSO': {MODELO_SOSPECHOSO}")
        print(f"Error: {e}")
        return
    
    print("✅ Modelos cargados")
    
    # Inicializar
    tracker_general = SimpleTracker()
    tracker_armas = SimpleTracker()
    behavior_analyzer = BehaviorAnalyzer()
    
    # Captura de video
    source_input = 0 if video_source == '0' else video_source
    cap = cv2.VideoCapture(source_input)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir la fuente de video: {video_source}")
        return
    
    frame_count = 0
    
    print("\n🚀 Sistema iniciado. Presiona 'ESC' para salir.")
    print("━" * 60)
    
    alert_history = []
    
    try:
        while True:
            ret, frame = cap.read()
                
            if not ret:
                print("...video finalizado.")
                break
            
            frame_display = frame.copy()
            
            # ═══════════════════════════════════════════════════
            # DETECCIÓN
            # ═══════════════════════════════════════════════════
            resultados_generales = modelo_general(frame, verbose=False)[0]
            resultados_armas = modelo_armas(frame, verbose=False)[0]
            resultados_sospechosos = modelo_sospechoso(frame, verbose=False)[0]
            
            
            # Convertir con umbral ajustado
            detecciones_generales = []
            for r in resultados_generales.boxes.data.cpu().numpy():
                if int(r[5]) == 0 and r[4] > CONFIDENCE_GENERAL:
                    detecciones_generales.append(r)
            
            detecciones_armas = []
            for r in resultados_armas.boxes.data.cpu().numpy():
                if r[4] > CONFIDENCE_ARMAS: # <-- Umbral alto (0.7) se aplica aquí
                    detecciones_armas.append(r)
            
            detecciones_sospechosas = []
            for r in resultados_sospechosos.boxes.data.cpu().numpy():
                if r[4] > CONFIDENCE_SOSPECHOSO: # <-- Umbral (0.5) se aplica aquí
                    detecciones_sospechosas.append(r)
            
            
            # DEBUG: Mostrar detecciones de armas
            if len(detecciones_armas) > 0:
                print(f"⚠️  Frame {frame_count}: {len(detecciones_armas)} arma(s) detectada(s)")
            
            # ═══════════════════════════════════════════════════
            # TRACKING
            # ═══════════════════════════════════════════════════
            tracks_general = tracker_general.update(detecciones_generales)
            tracks_armas = tracker_armas.update(detecciones_armas)
            
            # ═══════════════════════════════════════════════════
            # ANÁLISIS DE COMPORTAMIENTO (MOVIMIENTO)
            # ═══════════════════════════════════════════════════
            alertas_activas = []
            
            for track_id, bbox in tracks_general.items():
                trajectory = list(tracker_general.trajectories[track_id])
                
                # Usamos el len() de la clase BehaviorAnalyzer (ahora es 5)
                if len(trajectory) >= 5: 
                    weapon_nearby = False
                    for arma_id, arma_bbox in tracks_armas.items():
                        overlap = bbox_overlap(bbox, arma_bbox)
                        distance = bbox_distance(bbox, arma_bbox)
                        
                        if overlap > 0.1 or distance < 100:
                            weapon_nearby = True
                            print(f"ALERTA: ARMA CERCA de ID:{track_id} (overlap={overlap:.2f}, dist={distance:.1f})")
                            break
                    
                    behavior, alert_level, features = behavior_analyzer.analyze_trajectory(
                        trajectory, track_id, weapon_nearby
                    )
                    
                    if alert_level > 0:
                        alertas_activas.append({
                            'track_id': track_id,
                            'behavior': behavior,
                            'alert_level': alert_level,
                            'bbox': bbox,
                            'features': features
                        })
            
            # ═══════════════════════════════════════════════════
            # VISUALIZACIÓN
            # ═══════════════════════════════════════════════════
            
            # Dibujar detecciones generales (personas)
            for track_id, bbox in tracks_general.items():
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame_display, f'Persona ID:{track_id}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
            # Dibujar alertas de MOVIMIENTO (Sin emojis)
            for alerta in alertas_activas:
                x1, y1, x2, y2 = alerta['bbox']
                
                if alerta['alert_level'] == 3:
                    color = (0, 0, 255)
                    label = f"ALERTA MAX: {alerta['behavior'].upper()}"
                elif alerta['alert_level'] == 2:
                    color = (0, 165, 255)
                    label = f"ALERTA: {alerta['behavior'].upper()}"
                else:
                    color = (0, 255, 255)
                    label = f"AVISO: {alerta['behavior'].upper()}"
                
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv2.putText(frame_display, label, (int(x1), int(y1)-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                vel = alerta['features'].get('velocity_mean', 0)
                dwell = alerta['features'].get('dwelling_time', 0)
                info = f"Vel:{vel:.1f} Tiempo:{dwell:.1f}s"
                cv2.putText(frame_display, info, (int(x1), int(y2)+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                alert_history.append({
                    'frame': frame_count,
                    'time': time.time(),
                    'track_id': alerta['track_id'],
                    'behavior': alerta['behavior'],
                    'level': alerta['alert_level']
                })

            
            # Dibujar armas
            for track_id, bbox in tracks_armas.items():
                x1, y1, x2, y2 = bbox
                color = (0, 0, 255) 
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
                cv2.putText(frame_display, 'ARMA DETECTADA', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
                
                center = (int((x1+x2)/2), int((y1+y2)/2))
                radius = int(max(x2-x1, y2-y1) / 2) + 10
                cv2.circle(frame_display, center, radius, color, 3)

            
            # Dibujar detecciones de COMPORTAMIENTOS
            for det in detecciones_sospechosas:
                x1, y1, x2, y2, conf, cls = det
                class_id = int(cls)
                
                class_name = get_class_name(class_id, is_weapon=False)
                color = get_class_color(class_name, is_weapon=False)
                risk_level = get_risk_level(class_id, is_weapon=False)

                if class_name == 'Unknown' or risk_level == 0:
                    continue 

                label = f"{class_name.upper()} ({conf:.2f})"
                
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(frame_display, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                if risk_level > 0 and class_name not in [a['behavior'] for a in alert_history]:
                     alert_history.append({
                        'frame': frame_count,
                        'time': time.time(),
                        'track_id': 'N/A',
                        'behavior': class_name,
                        'level': risk_level
                    })

            
            # Info general
            info_text = f"Tracks: {len(tracks_general)} | Armas: {len(detecciones_armas)} | Alertas: {len(alertas_activas)} | Frame: {frame_count}"
            cv2.putText(frame_display, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar frame
            cv2.imshow('NeuroEvol-Threat - Análisis Temporal Completo', frame_display)
            
            # Control de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # Guardar screenshot
                screenshot_path = f'screenshot_{frame_count}.png'
                cv2.imwrite(screenshot_path, frame_display)
                print(f"📸 Screenshot guardado: {screenshot_path}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupción del usuario")
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Reporte final
        print("\n" + "━" * 60)
        print("📊 REPORTE FINAL")
        print("━" * 60)
        print(f"Frames procesados: {frame_count}")
        
        unique_alerts = []
        last_behaviors = {}
        for alert in alert_history:
            behavior = alert['behavior']
            current_time = alert['time']
            if behavior not in last_behaviors or (current_time - last_behaviors[behavior] > 5):
                unique_alerts.append(alert)
                last_behaviors[behavior] = current_time

        print(f"Total de alertas únicas: {len(unique_alerts)}")
        
        if unique_alerts:
            print("\nAlertas por tipo:")
            from collections import Counter
            behavior_counts = Counter([a['behavior'] for a in unique_alerts])
            for behavior, count in behavior_counts.most_common():
                print(f"  {behavior:20s}: {count}")
        
        print("━" * 60)
        print("✅ Sistema finalizado")


if __name__ == "__main__":
    print("--- BLOQUE DE EJECUCIÓN PRINCIPAL INICIADO ---") 
    parser = argparse.ArgumentParser(description='NeuroEvol-Threat - Sistema de Análisis Temporal')
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Fuente de video (0 para webcam, o ruta a archivo)'
    )
    
    parser.add_argument(
        '--lstm-model',
        type=str,
        default=None,
        help='Ruta al modelo LSTM entrenado (.pth)'
    )
    
    parser.add_argument(
        '--confidence-general',
        type=float,
        default=0.3,
        help='Umbral de confianza para detección general'
    )

    parser.add_argument(
        '--confidence-behavior',
        type=float,
        default=0.5, # <-- ### NUEVA CORRECCIÓN ###
        help='Umbral de confianza para comportamientos'
    )
    
    parser.add_argument(
        '--confidence-weapon',
        type=float,
        default=0.70, # <-- Mantenemos alto
        help='Umbral de confianza para armas'
    )
    
    args = parser.parse_args()
    
    # Actualizar variables globales
    CONFIDENCE_GENERAL = args.confidence_general
    CONFIDENCE_ARMAS = args.confidence_weapon
    CONFIDENCE_SOSPECHOSO = args.confidence_behavior 
    
    import multiprocessing
    
    # multiprocessing.freeze_support() 
    
    print("--- LLAMANDO A LA FUNCIÓN main() ---") 
    main(args.source)