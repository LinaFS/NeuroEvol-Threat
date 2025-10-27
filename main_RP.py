"""
main.py - VERSIÓN CON RECONOCIMIENTO DE PATRONES TEMPORALES

Este archivo REEMPLAZA tu main.py actual e integra:
1. Detección YOLOv8 (optimizada con GA)
2. Tracking multi-objeto
3. Análisis temporal de comportamientos
4. Sistema de alertas
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque, defaultdict

# ============================================
# CONFIGURACIÓN
# ============================================
MODELO_GENERAL = 'yolov8n.pt'
MODELO_SOSPECHOSO = 'ModeloSospechaOptimizado/best_model_ga_optimized/weights/best.pt'
MODELO_ARMAS = 'ModeloArmasOptimizado/best_model_ga_optimized/weights/best.pt'

# Parámetros de tracking
MAX_DISAPPEARED = 30  # Frames máximos sin detección
DISTANCE_THRESHOLD = 50  # Píxeles máximos para asociar detección

# Parámetros de análisis temporal
WINDOW_SIZE = 30  # Frames para analizar (1 segundo a 30 FPS)
LOITERING_TIME = 10  # Segundos para considerar "merodeando"
VELOCITY_THRESHOLD = 5  # px/frame para considerar "estático"

# ============================================
# CLASE: SIMPLE TRACKER (Sin dependencias extras)
# ============================================
class SimpleTracker:
    """
    Tracker simple basado en distancia euclidiana
    No requiere DeepSORT ni dependencias adicionales
    """
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}  # {id: centroid}
        self.disappeared = {}  # {id: frames_sin_deteccion}
        self.trajectories = defaultdict(lambda: deque(maxlen=90))  # 3 seg historial
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
        # Mantener trayectoria para análisis posterior
    
    def update(self, detections):
        """
        Actualizar tracker con nuevas detecciones
        detections: List[(x1, y1, x2, y2, confidence, class)]
        
        Returns: Dict[id: (x1, y1, x2, y2, class)]
        """
        if len(detections) == 0:
            # Incrementar contador de desaparecidos
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # Calcular centroides de detecciones
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
        
        # Si no hay objetos tracked, registrar todos
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i])
        else:
            # Asociar detecciones con objetos existentes
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calcular distancias
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i, j] = np.linalg.norm(
                        np.array(obj_centroid) - np.array(input_centroid)
                    )
            
            # Asociación simple: mínima distancia
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if distances[row, col] > DISTANCE_THRESHOLD:
                    continue
                
                # Actualizar objeto
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
            
            # Objetos no asociados → desaparecidos
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Detecciones no asociadas → nuevos objetos
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])
        
        # Retornar objetos activos con sus bboxes
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
    """
    Analiza trayectorias para detectar comportamientos sospechosos
    """
    def __init__(self):
        self.alert_cooldown = {}  # {track_id: last_alert_time}
        self.cooldown_time = 10  # segundos entre alertas
    
    def analyze_trajectory(self, trajectory, track_id, weapon_detected=False):
        """
        Analizar trayectoria y determinar comportamiento
        
        Returns:
        - behavior: str ('normal', 'loitering', 'erratic', 'weapon_carry')
        - alert_level: int (0=normal, 1=bajo, 2=medio, 3=alto)
        - features: dict con características calculadas
        """
        if len(trajectory) < 5:
            return 'normal', 0, {}
        
        # Extraer características
        features = self._extract_features(trajectory)
        
        # Reglas de detección
        behavior = 'normal'
        alert_level = 0
        
        # 1. PORTACIÓN DE ARMA (Prioridad máxima)
        if weapon_detected:
            behavior = 'weapon_carry'
            alert_level = 3
        
        # 2. LOITERING (Merodeando)
        elif features['dwelling_time'] > LOITERING_TIME and features['velocity_mean'] < 1.0:
            behavior = 'loitering'
            alert_level = 2
        
        # 3. MOVIMIENTO ERRÁTICO
        elif features['direction_changes'] > 7 and features['velocity_std'] > 3.0:
            behavior = 'erratic_movement'
            alert_level = 2
        
        # 4. VELOCIDAD ANORMAL (Corriendo)
        elif features['velocity_mean'] > 10.0:
            behavior = 'running'
            alert_level = 1
        
        # Verificar cooldown para no spamear alertas
        if alert_level > 0:
            current_time = time.time()
            if track_id in self.alert_cooldown:
                if current_time - self.alert_cooldown[track_id] < self.cooldown_time:
                    alert_level = 0  # Suprimir alerta
            self.alert_cooldown[track_id] = current_time
        
        return behavior, alert_level, features
    
    def _extract_features(self, trajectory):
        """Extraer características de la trayectoria"""
        features = {}
        
        # Calcular velocidades
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
        
        # Tiempo de permanencia (dwelling time)
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
        
        # Distancia total recorrida
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
def main():
    # Cargar modelos
    print("🔧 Cargando modelos...")
    modelo_general = YOLO(MODELO_GENERAL)
    modelo_armas = YOLO(MODELO_ARMAS)
    print("✅ Modelos cargados")
    
    # Inicializar tracker y analizador
    tracker_general = SimpleTracker()
    tracker_armas = SimpleTracker()
    behavior_analyzer = BehaviorAnalyzer()
    
    # Captura de video
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    
    print("\n🚀 Sistema iniciado. Presiona 'ESC' para salir.")
    print("━" * 60)
    
    # Estadísticas
    alert_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_display = frame.copy()
            
            # ═══════════════════════════════════════════════════
            # PASO 1: DETECCIÓN CON YOLO
            # ═══════════════════════════════════════════════════
            resultados_generales = modelo_general(frame, verbose=False)[0]
            resultados_armas = modelo_armas(frame, verbose=False)[0]
            
            # Convertir resultados a formato [x1, y1, x2, y2, conf, class]
            detecciones_generales = []
            for r in resultados_generales.boxes.data.cpu().numpy():
                if r[4] > 0.5:  # Confianza > 0.5
                    detecciones_generales.append(r)
            
            detecciones_armas = []
            for r in resultados_armas.boxes.data.cpu().numpy():
                if r[4] > 0.5:
                    detecciones_armas.append(r)
            
            # ═══════════════════════════════════════════════════
            # PASO 2: TRACKING (Asignar IDs persistentes)
            # ═══════════════════════════════════════════════════
            tracks_general = tracker_general.update(detecciones_generales)
            tracks_armas = tracker_armas.update(detecciones_armas)
            
            # ═══════════════════════════════════════════════════
            # PASO 3: ANÁLISIS DE COMPORTAMIENTO
            # ═══════════════════════════════════════════════════
            alertas_activas = []
            
            for track_id, bbox in tracks_general.items():
                trajectory = list(tracker_general.trajectories[track_id])
                
                if len(trajectory) >= WINDOW_SIZE:
                    # Verificar si hay arma detectada cerca
                    weapon_nearby = any(
                        self._bbox_overlap(bbox, arma_bbox) > 0.3
                        for arma_bbox in tracks_armas.values()
                    )
                    
                    # Analizar comportamiento
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
            # PASO 4: VISUALIZACIÓN
            # ═══════════════════════════════════════════════════
            
            # Dibujar detecciones generales (verde)
            for track_id, bbox in tracks_general.items():
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)  # Verde
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame_display, f'ID:{track_id}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Dibujar alertas (colores según nivel)
            for alerta in alertas_activas:
                x1, y1, x2, y2 = alerta['bbox']
                
                # Color según nivel de alerta
                if alerta['alert_level'] == 3:
                    color = (0, 0, 255)  # Rojo
                    label = f"🔴 {alerta['behavior'].upper()}"
                elif alerta['alert_level'] == 2:
                    color = (0, 165, 255)  # Naranja
                    label = f"🟠 {alerta['behavior'].upper()}"
                else:
                    color = (0, 255, 255)  # Amarillo
                    label = f"🟡 {alerta['behavior'].upper()}"
                
                # Bbox más grueso para alertas
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                
                # Texto de alerta
                cv2.putText(frame_display, label, (int(x1), int(y1)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Mostrar características clave
                vel = alerta['features'].get('velocity_mean', 0)
                dwell = alerta['features'].get('dwelling_time', 0)
                info = f"Vel:{vel:.1f} Tiempo:{dwell:.1f}s"
                cv2.putText(frame_display, info, (int(x1), int(y2)+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Guardar en historial
                alert_history.append({
                    'frame': frame_count,
                    'time': time.time(),
                    'track_id': alerta['track_id'],
                    'behavior': alerta['behavior'],
                    'level': alerta['alert_level']
                })
            
            # Dibujar armas detectadas (rojo intenso)
            for track_id, bbox in tracks_armas.items():
                x1, y1, x2, y2 = bbox
                color = (0, 0, 255)  # Rojo
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(frame_display, 'ARMA', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Info general
            info_text = f"Tracks: {len(tracks_general)} | Alertas: {len(alertas_activas)} | Frame: {frame_count}"
            cv2.putText(frame_display, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mostrar resultado
            cv2.imshow('NeuroEvol-Threat - Reconocimiento de Patrones', frame_display)
            
            # Control
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Reporte final
        print("\n" + "━" * 60)
        print("📊 REPORTE FINAL")
        print("━" * 60)
        print(f"Frames procesados: {frame_count}")
        print(f"Total de alertas: {len(alert_history)}")
        
        if alert_history:
            print("\nAlertas por tipo:")
            from collections import Counter
            behavior_counts = Counter([a['behavior'] for a in alert_history])
            for behavior, count in behavior_counts.most_common():
                print(f"  {behavior:20s}: {count}")
        
        print("━" * 60)
        print("✅ Sistema finalizado")

def _bbox_overlap(bbox1, bbox2):
    """Calcular IoU entre dos bboxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Intersección
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Áreas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # IoU
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()