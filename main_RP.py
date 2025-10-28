"""
main.py - SISTEMA MULTI-CLASE INTELIGENTE
Mejoras:
- Detecta 80 clases COCO (no solo personas)
- Identifica objetos de riesgo (mochilas, cuchillos, vehículos)
- Detecta asociaciones persona-objeto
- Detecta objetos abandonados
- Análisis contextual de riesgo
"""

print("--- SISTEMA MULTI-CLASE INICIADO ---")

from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque, defaultdict
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
)

# ============================================
# CONFIGURACIÓN
# ============================================
MODELO_GENERAL = 'yolo11n.pt'
MODELO_SOSPECHOSO = r'ModeloSopechaOptimizado\best_model_ga_optimized\weights\best.pt'
MODELO_ARMAS = r'ModeloArmasOptimizado\best_model_ga_optimized\weights\best.pt'

# ============================================
# CLASES COCO DE INTERÉS PARA SEGURIDAD
# ============================================
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 24: 'backpack',
    26: 'handbag', 28: 'suitcase', 43: 'knife', 76: 'scissors',
    39: 'bottle', 67: 'cell phone'
}

# Clasificación de objetos por riesgo
OBJETOS_ALTO_RIESGO = [43, 76]  # knife, scissors
OBJETOS_SOSPECHOSOS = [24, 26, 28]  # backpack, handbag, suitcase
VEHICULOS = [1, 2, 3, 5, 6, 7, 8]  # bicycle, car, motorcycle, bus, etc.
OBJETOS_VIGILANCIA = [67, 39]  # cell phone, bottle

# Colores por categoría
COLOR_PERSONA = (0, 255, 0)  # Verde
COLOR_ALTO_RIESGO = (0, 0, 255)  # Rojo
COLOR_SOSPECHOSO = (0, 165, 255)  # Naranja
COLOR_VEHICULO = (255, 165, 0)  # Azul-Naranja
COLOR_NORMAL = (200, 200, 200)  # Gris

# Parámetros
MAX_DISAPPEARED = 30
DISTANCE_THRESHOLD = 50
LOITERING_TIME = 10
CONFIDENCE_GENERAL = 0.3
CONFIDENCE_SOSPECHOSO = 0.5
CONFIDENCE_ARMAS = 0.70

# Parámetros para objetos abandonados
ABANDONED_TIME_THRESHOLD = 15  # segundos
PERSON_PROXIMITY_THRESHOLD = 150  # píxeles

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def bbox_overlap(bbox1, bbox2):
    """Calcular IoU entre dos bboxes"""
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
    """Calcular distancia entre centroides"""
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def get_class_info(class_id):
    """Obtener información de la clase"""
    class_id = int(class_id)
    name = COCO_CLASSES.get(class_id, f'Object_{class_id}')
    
    # Determinar color y nivel de riesgo
    if class_id == 0:
        color = COLOR_PERSONA
        risk = 0
    elif class_id in OBJETOS_ALTO_RIESGO:
        color = COLOR_ALTO_RIESGO
        risk = 3
    elif class_id in OBJETOS_SOSPECHOSOS:
        color = COLOR_SOSPECHOSO
        risk = 2
    elif class_id in VEHICULOS:
        color = COLOR_VEHICULO
        risk = 1
    else:
        color = COLOR_NORMAL
        risk = 0
    
    return name, color, risk


# ============================================
# CLASE: MULTI-CLASS TRACKER
# ============================================
class MultiClassTracker:
    """Tracker que maneja múltiples clases de objetos"""
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}  # id: centroid
        self.disappeared = {}  # id: count
        self.trajectories = defaultdict(lambda: deque(maxlen=90))
        self.class_history = {}  # id: class_id
        self.first_seen = {}  # id: timestamp
        self.max_disappeared = max_disappeared
        
    def register(self, centroid, bbox, class_id):
        """Registrar nuevo objeto"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.class_history[self.next_id] = class_id
        self.first_seen[self.next_id] = time.time()
        self.trajectories[self.next_id].append({
            'centroid': centroid,
            'bbox': bbox,
            'timestamp': time.time(),
            'class_id': class_id
        })
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id):
        """Eliminar objeto"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.class_history:
            del self.class_history[object_id]
        if object_id in self.first_seen:
            del self.first_seen[object_id]
    
    def update(self, detections):
        """
        Actualizar tracker
        detections: List[(x1, y1, x2, y2, confidence, class)]
        Returns: Dict[id: (bbox, class_id, time_tracked)]
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
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
                self.register(centroid, input_bboxes[i], input_classes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
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
                self.class_history[object_id] = input_classes[col]
                self.trajectories[object_id].append({
                    'centroid': input_centroids[col],
                    'bbox': input_bboxes[col],
                    'timestamp': time.time(),
                    'class_id': input_classes[col]
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
                self.register(input_centroids[col], input_bboxes[col], input_classes[col])
        
        # Retornar objetos activos con información completa
        active_objects = {}
        current_time = time.time()
        for object_id in self.objects.keys():
            if len(self.trajectories[object_id]) > 0:
                last_point = self.trajectories[object_id][-1]
                time_tracked = current_time - self.first_seen[object_id]
                active_objects[object_id] = (
                    last_point['bbox'], 
                    self.class_history.get(object_id, 0),
                    time_tracked
                )
        
        return active_objects


# ============================================
# CLASE: ANALIZADOR DE COMPORTAMIENTO
# ============================================
class BehaviorAnalyzer:
    """Analiza trayectorias de personas"""
    def __init__(self):
        self.alert_cooldown = {}
        self.cooldown_time = 5
    
    def analyze_trajectory(self, trajectory, track_id, weapon_detected=False):
        if len(trajectory) < 5: 
            return 'normal', 0, {}
        
        features = self._extract_features(trajectory)
        behavior = 'normal'
        alert_level = 0
        
        if weapon_detected:
            behavior = 'weapon_carry'
            alert_level = 3
        elif features['dwelling_time'] > LOITERING_TIME and features['velocity_mean'] < 0.5:
            behavior = 'loitering'
            alert_level = 2
        elif features['direction_changes'] > 8 and features['velocity_std'] > 3.0:
            behavior = 'erratic_movement'
            alert_level = 2
        elif features['velocity_mean'] > 12.0:
            behavior = 'running'
            alert_level = 1
        
        if alert_level > 0:
            current_time = time.time()
            if track_id in self.alert_cooldown:
                if current_time - self.alert_cooldown[track_id] < self.cooldown_time:
                    alert_level = 0
            self.alert_cooldown[track_id] = current_time
        
        return behavior, alert_level, features
    
    def _extract_features(self, trajectory):
        features = {}
        
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
        
        total_time = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
        features['dwelling_time'] = total_time
        
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
        
        return features


# ============================================
# CLASE: ANALIZADOR CONTEXTUAL
# ============================================
class ContextAnalyzer:
    """Analiza relaciones entre personas y objetos"""
    
    @staticmethod
    def detect_abandoned_objects(object_tracks, person_tracks):
        """Detectar objetos abandonados (sin personas cerca)"""
        abandoned = []
        
        for obj_id, (obj_bbox, obj_class, time_tracked) in object_tracks.items():
            # Solo objetos sospechosos
            if obj_class not in OBJETOS_SOSPECHOSOS:
                continue
            
            # Verificar tiempo mínimo
            if time_tracked < ABANDONED_TIME_THRESHOLD:
                continue
            
            # Verificar si hay personas cerca
            has_person_nearby = False
            for person_id, (person_bbox, person_class, _) in person_tracks.items():
                if person_class != 0:  # Solo personas
                    continue
                
                distance = bbox_distance(obj_bbox, person_bbox)
                if distance < PERSON_PROXIMITY_THRESHOLD:
                    has_person_nearby = True
                    break
            
            if not has_person_nearby:
                abandoned.append({
                    'id': obj_id,
                    'bbox': obj_bbox,
                    'class': obj_class,
                    'time': time_tracked
                })
        
        return abandoned
    
    @staticmethod
    def detect_person_object_associations(person_tracks, object_tracks):
        """Detectar personas con objetos de riesgo"""
        associations = []
        
        for person_id, (person_bbox, person_class, _) in person_tracks.items():
            if person_class != 0:
                continue
            
            for obj_id, (obj_bbox, obj_class, _) in object_tracks.items():
                if obj_class in OBJETOS_SOSPECHOSOS + OBJETOS_ALTO_RIESGO:
                    distance = bbox_distance(person_bbox, obj_bbox)
                    overlap = bbox_overlap(person_bbox, obj_bbox)
                    
                    if distance < 100 or overlap > 0.1:
                        risk_level = 3 if obj_class in OBJETOS_ALTO_RIESGO else 2
                        associations.append({
                            'person_id': person_id,
                            'object_id': obj_id,
                            'object_class': obj_class,
                            'distance': distance,
                            'risk_level': risk_level
                        })
        
        return associations


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main(video_source):
    print("🔧 Cargando modelos...")
    modelo_general = YOLO(MODELO_GENERAL)
    modelo_armas = YOLO(MODELO_ARMAS)
    
    try:
        modelo_sospechoso = YOLO(MODELO_SOSPECHOSO)
    except Exception as e:
        print(f"⚠️  Modelo de comportamientos no disponible: {e}")
        modelo_sospechoso = None
    
    print("✅ Modelos cargados")
    
    # Inicializar trackers (separados por tipo)
    tracker_objetos = MultiClassTracker()
    tracker_armas = MultiClassTracker()
    behavior_analyzer = BehaviorAnalyzer()
    context_analyzer = ContextAnalyzer()
    
    # Captura de video
    source_input = 0 if video_source == '0' else video_source
    cap = cv2.VideoCapture(source_input)
    
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video: {video_source}")
        return
    
    frame_count = 0
    alert_history = []
    
    print("\n🚀 Sistema Multi-Clase Iniciado")
    print("━" * 60)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_display = frame.copy()
            
            # ═══════════════════════════════════════════════════
            # DETECCIÓN
            # ═══════════════════════════════════════════════════
            resultados_generales = modelo_general(frame, verbose=False)[0]
            resultados_armas = modelo_armas(frame, verbose=False)[0]
            
            # CAMBIO CRÍTICO: Ya NO filtramos solo personas
            detecciones_objetos = []
            for r in resultados_generales.boxes.data.cpu().numpy():
                if r[4] > CONFIDENCE_GENERAL:  # Sin filtro de clase
                    detecciones_objetos.append(r)
            
            detecciones_armas = []
            for r in resultados_armas.boxes.data.cpu().numpy():
                if r[4] > CONFIDENCE_ARMAS:
                    detecciones_armas.append(r)
            
            detecciones_sospechosas = []
            if modelo_sospechoso:
                resultados_sospechosos = modelo_sospechoso(frame, verbose=False)[0]
                for r in resultados_sospechosos.boxes.data.cpu().numpy():
                    if r[4] > CONFIDENCE_SOSPECHOSO:
                        detecciones_sospechosas.append(r)
            
            # ═══════════════════════════════════════════════════
            # TRACKING
            # ═══════════════════════════════════════════════════
            tracks_objetos = tracker_objetos.update(detecciones_objetos)
            tracks_armas = tracker_armas.update(detecciones_armas)
            
            # Separar personas de otros objetos
            tracks_personas = {k: v for k, v in tracks_objetos.items() if v[1] == 0}
            tracks_otros = {k: v for k, v in tracks_objetos.items() if v[1] != 0}
            
            # ═══════════════════════════════════════════════════
            # ANÁLISIS CONTEXTUAL
            # ═══════════════════════════════════════════════════
            objetos_abandonados = context_analyzer.detect_abandoned_objects(
                tracks_otros, tracks_personas
            )
            
            asociaciones = context_analyzer.detect_person_object_associations(
                tracks_personas, {**tracks_otros, **tracks_armas}
            )
            
            # ═══════════════════════════════════════════════════
            # ANÁLISIS DE COMPORTAMIENTO (PERSONAS)
            # ═══════════════════════════════════════════════════
            alertas_comportamiento = []
            
            for track_id, (bbox, class_id, _) in tracks_personas.items():
                trajectory = list(tracker_objetos.trajectories[track_id])
                
                if len(trajectory) >= 5:
                    # Verificar si tiene arma cerca
                    weapon_nearby = any(
                        bbox_distance(bbox, arma_bbox) < 100
                        for _, (arma_bbox, _, _) in tracks_armas.items()
                    )
                    
                    behavior, alert_level, features = behavior_analyzer.analyze_trajectory(
                        trajectory, track_id, weapon_nearby
                    )
                    
                    if alert_level > 0:
                        alertas_comportamiento.append({
                            'track_id': track_id,
                            'behavior': behavior,
                            'alert_level': alert_level,
                            'bbox': bbox,
                            'features': features
                        })
            
            # ═══════════════════════════════════════════════════
            # VISUALIZACIÓN
            # ═══════════════════════════════════════════════════
            
            # 1. Dibujar todos los objetos detectados
            for track_id, (bbox, class_id, time_tracked) in tracks_objetos.items():
                x1, y1, x2, y2 = bbox
                class_name, color, risk = get_class_info(class_id)
                
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f'{class_name} ID:{track_id}'
                cv2.putText(frame_display, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 2. Dibujar armas
            for track_id, (bbox, class_id, _) in tracks_armas.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                cv2.putText(frame_display, 'ARMA DETECTADA', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            
            # 3. Dibujar objetos abandonados
            for obj in objetos_abandonados:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
                cv2.putText(frame_display, f'OBJETO ABANDONADO ({obj["time"]:.1f}s)', 
                            (int(x1), int(y1)-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 4. Dibujar asociaciones persona-objeto
            for assoc in asociaciones:
                person_bbox = tracks_personas[assoc['person_id']][0]
                obj_name, _, _ = get_class_info(assoc['object_class'])
                
                x1, y1, x2, y2 = person_bbox
                color = (0, 0, 255) if assoc['risk_level'] == 3 else (0, 165, 255)
                label = f"ALERTA: {obj_name.upper()}"
                cv2.putText(frame_display, label, (int(x1), int(y1)-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 5. Dibujar alertas de comportamiento
            for alerta in alertas_comportamiento:
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
                cv2.putText(frame_display, label, (int(x1), int(y1)-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 6. Dibujar comportamientos sospechosos (modelo custom)
            for det in detecciones_sospechosas:
                x1, y1, x2, y2, conf, cls = det
                class_id = int(cls)
                
                class_name = get_class_name(class_id, is_weapon=False)
                color = get_class_color(class_name, is_weapon=False)
                risk_level = get_risk_level(class_id, is_weapon=False)
                
                if class_name != 'Unknown' and risk_level > 0:
                    label = f"{class_name.upper()} ({conf:.2f})"
                    cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(frame_display, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Info general
            info_text = f"Objetos: {len(tracks_objetos)} | Personas: {len(tracks_personas)} | Armas: {len(tracks_armas)} | Abandonados: {len(objetos_abandonados)}"
            cv2.putText(frame_display, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Sistema Multi-Clase Inteligente', frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_{frame_count}.png', frame_display)
                print(f"📸 Screenshot guardado")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupción del usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Sistema finalizado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sistema Multi-Clase Inteligente')
    parser.add_argument('--source', type=str, default='0', help='Fuente de video')
    parser.add_argument('--confidence-general', type=float, default=0.3)
    parser.add_argument('--confidence-behavior', type=float, default=0.5)
    parser.add_argument('--confidence-weapon', type=float, default=0.70)
    
    args = parser.parse_args()
    
    # Actualizar variables globales
    CONFIDENCE_GENERAL = args.confidence_general
    CONFIDENCE_SOSPECHOSO = args.confidence_behavior
    CONFIDENCE_ARMAS = args.confidence_weapon
    
    print("--- SISTEMA MULTI-CLASE INICIADO ---")
    main(args.source)