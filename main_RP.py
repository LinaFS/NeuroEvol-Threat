"""
main_temporal.py - Sistema Completo con LSTM y Clases Reales
Integrado con classes_config.py para usar las 21 clases de comportamiento
"""

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
# CONFIGURACIÓN
# ============================================
class Config:
    # Modelos
    MODELO_GENERAL = 'yolov8n.pt'
    MODELO_SOSPECHOSO = 'ModeloSospechaOptimizado/best_model_ga_optimized/weights/best.pt'
    MODELO_ARMAS = 'ModeloArmasOptimizado/best_model_ga_optimized/weights/best.pt'
    MODELO_LSTM = 'models/behavior_lstm_final.pth'
    
    # Tracking
    MAX_DISAPPEARED = 30
    DISTANCE_THRESHOLD = 50
    
    # Análisis temporal
    WINDOW_SIZE = 30
    MIN_TRAJECTORY_LENGTH = 5
    
    # Umbrales de confianza
    CONFIDENCE_GENERAL = 0.3
    CONFIDENCE_SOSPECHOSO = 0.4
    CONFIDENCE_ARMAS = 0.25
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()


# ============================================
# MODELO LSTM
# ============================================
class BehaviorLSTM(nn.Module):
    """Modelo LSTM para clasificación de comportamientos temporales"""
    
    def __init__(self, input_dim=20, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super(BehaviorLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.batch_norm = nn.BatchNorm1d(hidden_dim // 2)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, hidden


# ============================================
# MAPEO DE CLASES LSTM A CLASES REALES
# ============================================
LSTM_TO_BEHAVIOR = {
    0: 'Normal_Videos',      # normal
    1: 'Meet_and_Split',     # loitering
    2: 'Assault',            # aggression
    3: 'Shooting',           # weapon_carry
    4: 'Stealing',           # erratic_movement
    5: 'Explosion'           # critical
}


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


# ============================================
# TRACKER
# ============================================
class SimpleTracker:
    """Tracker simple con historial de trayectorias"""
    
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.trajectories = defaultdict(lambda: deque(maxlen=90))
        self.class_history = defaultdict(lambda: deque(maxlen=30))
        self.max_disappeared = max_disappeared
        
    def register(self, centroid, bbox, class_id):
        """Registrar nuevo objeto"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trajectories[self.next_id].append({
            'centroid': centroid,
            'bbox': bbox,
            'timestamp': time.time(),
            'class': class_id
        })
        self.class_history[self.next_id].append(class_id)
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id):
        """Eliminar objeto perdido"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def get_dominant_class(self, object_id):
        """Obtener clase dominante en historial"""
        if object_id not in self.class_history or len(self.class_history[object_id]) == 0:
            return None
        
        # Retornar la clase más común en últimos frames
        from collections import Counter
        counts = Counter(self.class_history[object_id])
        return counts.most_common(1)[0][0]
    
    def update(self, detections):
        """
        Actualizar tracker
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
                self.register(centroid, input_bboxes[i], input_classes[i])
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
                
                if distances[row, col] > config.DISTANCE_THRESHOLD:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.trajectories[object_id].append({
                    'centroid': input_centroids[col],
                    'bbox': input_bboxes[col],
                    'timestamp': time.time(),
                    'class': input_classes[col]
                })
                self.class_history[object_id].append(input_classes[col])
                
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
        
        # Retornar objetos activos con su clase dominante
        active_objects = {}
        for object_id in self.objects.keys():
            if len(self.trajectories[object_id]) > 0:
                last_point = self.trajectories[object_id][-1]
                dominant_class = self.get_dominant_class(object_id)
                active_objects[object_id] = (*last_point['bbox'], dominant_class)
        
        return active_objects


# ============================================
# EXTRACTOR DE CARACTERÍSTICAS
# ============================================
class FeatureExtractor:
    """Extrae características temporales de trayectorias"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
    
    def extract(self, trajectory):
        """
        Extraer 20 características de una trayectoria
        Returns: np.array de shape (window_size, 20)
        """
        if len(trajectory) < 3:
            return None
        
        # Si la trayectoria es más corta que window_size, rellenar
        if len(trajectory) < self.window_size:
            # Repetir el primer punto
            padding = [trajectory[0]] * (self.window_size - len(trajectory))
            trajectory = padding + list(trajectory)
        else:
            # Tomar últimos window_size puntos
            trajectory = list(trajectory)[-self.window_size:]
        
        features_sequence = []
        
        for i in range(len(trajectory)):
            # Tomar ventana actual
            window_end = i + 1
            window = trajectory[max(0, window_end - 5):window_end]  # Ventana de 5 frames
            
            if len(window) < 2:
                # Si no hay suficientes datos, usar ceros
                features = np.zeros(20, dtype=np.float32)
            else:
                features = self._compute_features(window)
            
            features_sequence.append(features)
        
        return np.array(features_sequence, dtype=np.float32)
    
    def _compute_features(self, window):
        """Computar características de una ventana pequeña"""
        # Extraer datos
        centroids = np.array([p['centroid'] for p in window])
        timestamps = np.array([p['timestamp'] for p in window])
        
        # 1-2: Posición promedio
        x_mean = np.mean(centroids[:, 0]) / 640.0  # Normalizar por ancho típico
        y_mean = np.mean(centroids[:, 1]) / 480.0  # Normalizar por alto típico
        
        # 3-4: Desviación estándar de posición
        x_std = np.std(centroids[:, 0]) / 640.0
        y_std = np.std(centroids[:, 1]) / 480.0
        
        # 5: Área cubierta
        x_range = np.ptp(centroids[:, 0]) / 640.0
        y_range = np.ptp(centroids[:, 1]) / 480.0
        area_coverage = x_range * y_range
        
        # 6: Cercanía al borde
        near_edge = 1.0 if (x_mean < 0.1 or x_mean > 0.9 or y_mean < 0.1 or y_mean > 0.9) else 0.0
        
        # 7-9: Velocidades
        velocities = []
        for i in range(1, len(centroids)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = centroids[i][0] - centroids[i-1][0]
                dy = centroids[i][1] - centroids[i-1][1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        velocity_mean = np.mean(velocities) if velocities else 0.0
        velocity_max = np.max(velocities) if velocities else 0.0
        velocity_std = np.std(velocities) if velocities else 0.0
        
        # 10-11: Aceleraciones
        accelerations = []
        for i in range(1, len(velocities)):
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                acc = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(abs(acc))
        
        acceleration_mean = np.mean(accelerations) if accelerations else 0.0
        acceleration_max = np.max(accelerations) if accelerations else 0.0
        
        # 12: Cambios de dirección
        direction_changes = 0
        for i in range(2, len(centroids)):
            v1 = centroids[i-1] - centroids[i-2]
            v2 = centroids[i] - centroids[i-1]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                if angle > 45:
                    direction_changes += 1
        
        # 13: Tiempo de permanencia
        dwelling_time = timestamps[-1] - timestamps[0]
        
        # 14: Distancia total recorrida
        distance_traveled = 0
        for i in range(1, len(centroids)):
            distance_traveled += np.linalg.norm(centroids[i] - centroids[i-1])
        
        # 15: Duración de trayectoria
        trajectory_duration = dwelling_time
        
        # 16: Número de frames
        frames_count = len(window)
        
        # 17-20: Características adicionales (simplificadas)
        nearby_objects = 0.0  # Placeholder
        min_distance = 100.0  # Placeholder
        interaction_duration = 0.0  # Placeholder
        zone_visited = 0.0  # Placeholder
        
        return np.array([
            x_mean, y_mean, x_std, y_std, area_coverage, near_edge,
            velocity_mean, velocity_max, velocity_std,
            acceleration_mean, acceleration_max,
            direction_changes, dwelling_time, distance_traveled,
            trajectory_duration, frames_count,
            nearby_objects, min_distance, interaction_duration, zone_visited
        ], dtype=np.float32)


# ============================================
# ANALIZADOR DE COMPORTAMIENTO
# ============================================
class BehaviorAnalyzer:
    """Analiza comportamientos usando YOLO + LSTM + Reglas"""
    
    def __init__(self, lstm_model=None):
        self.lstm_model = lstm_model
        self.feature_extractor = FeatureExtractor(window_size=config.WINDOW_SIZE)
        self.alert_cooldown = {}
        self.cooldown_time = 5
        
        if self.lstm_model:
            self.lstm_model.eval()
            print("   ✅ Modelo LSTM cargado")
    
    def analyze(self, track_id, trajectory, yolo_class_id, weapon_detected=False):
        """
        Analizar comportamiento combinando YOLO + LSTM + Reglas
        
        Returns:
            behavior_name: str - Nombre del comportamiento
            alert_level: int - Nivel de alerta (0-3)
            confidence: float - Confianza de la predicción
            source: str - Fuente de la predicción (yolo/lstm/rule)
        """
        if len(trajectory) < config.MIN_TRAJECTORY_LENGTH:
            return 'normal', 0, 0.0, 'insufficient_data'
        
        # 1. Obtener clase de YOLO
        yolo_behavior = get_class_name(yolo_class_id, is_weapon=False)
        yolo_risk = get_risk_level(yolo_class_id, is_weapon=False)
        
        # 2. PRIORIDAD: Portación de arma
        if weapon_detected:
            return 'Shooting', 3, 1.0, 'weapon_detected'
        
        # 3. Si es clase crítica de YOLO, usar directamente
        if yolo_risk == 3:
            return yolo_behavior, yolo_risk, 0.9, 'yolo_critical'
        
        # 4. Si tenemos LSTM, usarlo para análisis temporal
        lstm_behavior = None
        lstm_confidence = 0.0
        
        if self.lstm_model and len(trajectory) >= 10:
            features = self.feature_extractor.extract(trajectory)
            if features is not None:
                try:
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(config.DEVICE)
                        outputs, _ = self.lstm_model(features_tensor)
                        probs = F.softmax(outputs, dim=1)
                        lstm_pred = torch.argmax(probs, dim=1).item()
                        lstm_confidence = probs[0, lstm_pred].item()
                        
                        # Mapear predicción LSTM a clase de comportamiento
                        lstm_behavior = LSTM_TO_BEHAVIOR.get(lstm_pred, 'Normal_Videos')
                except Exception as e:
                    print(f"   ⚠️  Error en LSTM: {e}")
        
        # 5. Decidir comportamiento final
        if lstm_behavior and lstm_confidence > 0.7:
            # Usar predicción LSTM si hay alta confianza
            final_behavior = lstm_behavior
            final_risk = get_risk_level(
                list(BEHAVIOR_CLASSES.keys())[list(BEHAVIOR_CLASSES.values()).index(lstm_behavior)],
                is_weapon=False
            )
            return final_behavior, final_risk, lstm_confidence, 'lstm'
        else:
            # Usar predicción de YOLO
            return yolo_behavior, yolo_risk, 0.8, 'yolo'
    
    def check_cooldown(self, track_id):
        """Verificar si el track está en cooldown"""
        if track_id in self.alert_cooldown:
            if time.time() - self.alert_cooldown[track_id] < self.cooldown_time:
                return True
        return False
    
    def set_cooldown(self, track_id):
        """Establecer cooldown para un track"""
        self.alert_cooldown[track_id] = time.time()


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main(args):
    """Función principal del sistema"""
    
    print("\n" + "="*70)
    print("🚀 NEUROEVOL-THREAT - Sistema de Análisis Temporal")
    print("="*70)
    
    # Configurar modelo LSTM si se proporciona
    lstm_model = None
    if args.lstm_model and Path(args.lstm_model).exists():
        print(f"\n📦 Cargando modelo LSTM: {args.lstm_model}")
        try:
            checkpoint = torch.load(args.lstm_model, map_location=config.DEVICE)
            model_config = checkpoint.get('config', {})
            
            lstm_model = BehaviorLSTM(
                input_dim=model_config.get('input_dim', 20),
                hidden_dim=model_config.get('hidden_dim', 128),
                num_layers=model_config.get('num_layers', 2),
                num_classes=model_config.get('num_classes', 6),
                dropout=model_config.get('dropout', 0.3)
            )
            lstm_model.load_state_dict(checkpoint['model_state_dict'])
            lstm_model.to(config.DEVICE)
            lstm_model.eval()
            print("   ✅ Modelo LSTM cargado correctamente")
        except Exception as e:
            print(f"   ⚠️  Error cargando LSTM: {e}")
            print("   ℹ️  Continuando sin análisis LSTM")
            lstm_model = None
    
    # Cargar modelos YOLO
    print("\n🔧 Cargando modelos YOLO...")
    modelo_general = YOLO(config.MODELO_GENERAL)
    modelo_sospechoso = YOLO(config.MODELO_SOSPECHOSO) if Path(config.MODELO_SOSPECHOSO).exists() else None
    modelo_armas = YOLO(config.MODELO_ARMAS) if Path(config.MODELO_ARMAS).exists() else None
    
    print(f"   ✅ Modelo general: {config.MODELO_GENERAL}")
    if modelo_sospechoso:
        print(f"   ✅ Modelo sospechoso: {config.MODELO_SOSPECHOSO}")
    if modelo_armas:
        print(f"   ✅ Modelo armas: {config.MODELO_ARMAS}")
    
    # Inicializar componentes
    tracker_general = SimpleTracker()
    tracker_sospechoso = SimpleTracker()
    tracker_armas = SimpleTracker()
    behavior_analyzer = BehaviorAnalyzer(lstm_model=lstm_model)
    
    # Captura de video
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la fuente de video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    
    print("\n" + "="*70)
    print("🎥 Sistema activo. Presiona 'ESC' para salir")
    print("="*70)
    
    alert_history = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_display = frame.copy()
            h, w = frame.shape[:2]
            
            # ═══════════════════════════════════════════════════
            # DETECCIÓN
            # ═══════════════════════════════════════════════════
            
            # Detección general
            resultados_generales = modelo_general(frame, verbose=False)[0]
            detecciones_generales = []
            for r in resultados_generales.boxes.data.cpu().numpy():
                if r[4] > config.CONFIDENCE_GENERAL and int(r[5]) == 0:  # Solo personas
                    detecciones_generales.append(r)
            
            # Detección de comportamientos sospechosos
            detecciones_sospechosas = []
            if modelo_sospechoso:
                resultados_sospechosos = modelo_sospechoso(frame, verbose=False)[0]
                for r in resultados_sospechosos.boxes.data.cpu().numpy():
                    if r[4] > config.CONFIDENCE_SOSPECHOSO:
                        detecciones_sospechosas.append(r)
            
            # Detección de armas
            detecciones_armas = []
            if modelo_armas:
                resultados_armas = modelo_armas(frame, verbose=False)[0]
                for r in resultados_armas.boxes.data.cpu().numpy():
                    if r[4] > config.CONFIDENCE_ARMAS:
                        detecciones_armas.append(r)
            
            # ═══════════════════════════════════════════════════
            # TRACKING
            # ═══════════════════════════════════════════════════
            tracks_general = tracker_general.update(detecciones_generales)
            tracks_sospechoso = tracker_sospechoso.update(detecciones_sospechosas)
            tracks_armas = tracker_armas.update(detecciones_armas)
            
            # ═══════════════════════════════════════════════════
            # ANÁLISIS DE COMPORTAMIENTO
            # ═══════════════════════════════════════════════════
            alertas_activas = []
            
            # Analizar tracks sospechosos
            for track_id, track_data in tracks_sospechoso.items():
                *bbox, class_id = track_data
                trajectory = list(tracker_sospechoso.trajectories[track_id])
                
                if len(trajectory) >= config.MIN_TRAJECTORY_LENGTH:
                    # Verificar arma cercana
                    weapon_nearby = False
                    for arma_id, arma_data in tracks_armas.items():
                        arma_bbox = arma_data[:4]
                        if bbox_overlap(bbox, arma_bbox) > 0.1 or bbox_distance(bbox, arma_bbox) < 100:
                            weapon_nearby = True
                            break
                    
                    # Analizar comportamiento
                    if not behavior_analyzer.check_cooldown(track_id):
                        behavior_name, alert_level, confidence, source = behavior_analyzer.analyze(
                            track_id, trajectory, class_id, weapon_nearby
                        )
                        
                        if alert_level > 0:
                            behavior_analyzer.set_cooldown(track_id)
                            alertas_activas.append({
                                'track_id': track_id,
                                'behavior': behavior_name,
                                'alert_level': alert_level,
                                'confidence': confidence,
                                'source': source,
                                'bbox': bbox,
                                'weapon_nearby': weapon_nearby
                            })
            
            # ═══════════════════════════════════════════════════
            # VISUALIZACIÓN
            # ═══════════════════════════════════════════════════
            
            # Dibujar detecciones generales (personas)
            for track_id, track_data in tracks_general.items():
                x1, y1, x2, y2, _ = track_data
                color = (0, 255, 0)
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            
            # Dibujar comportamientos sospechosos
            for track_id, track_data in tracks_sospechoso.items():
                x1, y1, x2, y2, class_id = track_data
                class_name = get_class_name(class_id, is_weapon=False)
                risk_level = get_risk_level(class_id, is_weapon=False)
                color = get_class_color(class_name, is_weapon=False)
                
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f'{class_name} ID:{track_id}'
                cv2.putText(frame_display, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Dibujar alertas
            for alerta in alertas_activas:
                x1, y1, x2, y2 = alerta['bbox']
                behavior = alerta['behavior']
                alert_level = alerta['alert_level']
                confidence = alerta['confidence']
                source = alerta['source']
                
                # Color según nivel de riesgo
                if alert_level == 3:
                    color = (0, 0, 255)  # Rojo
                    prefix = "🔴 CRÍTICO"
                elif alert_level == 2:
                    color = (0, 165, 255)  # Naranja
                    prefix = "🟠 ALERTA"
                else:
                    color = (0, 255, 255)  # Amarillo
                    prefix = "🟡 PRECAUCIÓN"
                
                # Dibujar bbox más grueso
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                
                # Etiqueta principal
                label = f"{prefix} - {behavior}"
                cv2.putText(frame_display, label, (int(x1), int(y1)-40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Info adicional
                info = f"ID:{alerta['track_id']} | Conf:{confidence:.2f} | {source}"
                cv2.putText(frame_display, info, (int(x1), int(y1)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Indicador de arma si aplica
                if alerta['weapon_nearby']:
                    cv2.putText(frame_display, "⚠️ ARMA DETECTADA", (int(x1), int(y2)+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Agregar a historial
                alert_history.append({
                    'frame': frame_count,
                    'time': time.time(),
                    'track_id': alerta['track_id'],
                    'behavior': behavior,
                    'level': alert_level,
                    'confidence': confidence,
                    'source': source
                })
            
            # Dibujar armas
            for track_id, track_data in tracks_armas.items():
                x1, y1, x2, y2, class_id = track_data
                weapon_name = get_class_name(class_id, is_weapon=True)
                color = (0, 0, 255)
                
                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
                cv2.putText(frame_display, f'⚠️ {weapon_name}', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Círculo pulsante
                center = (int((x1+x2)/2), int((y1+y2)/2))
                radius = int(max(x2-x1, y2-y1) / 2) + 10
                pulse = int(5 * np.sin(frame_count * 0.2)) + 10
                cv2.circle(frame_display, center, radius + pulse, color, 3)
            
            # Panel de información superior
            panel_height = 80
            overlay = frame_display.copy()
            cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame_display, 0.4, 0, frame_display)
            
            # Stats
            stats_text = [
                f"Frame: {frame_count}",
                f"Personas: {len(tracks_general)}",
                f"Sospechosos: {len(tracks_sospechoso)}",
                f"Armas: {len(tracks_armas)}",
                f"Alertas: {len(alertas_activas)}"
            ]
            
            x_offset = 10
            for i, text in enumerate(stats_text):
                cv2.putText(frame_display, text, (x_offset, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                x_offset += 150
            
            # Línea de separación
            cv2.line(frame_display, (0, panel_height), (w, panel_height), (255, 255, 255), 2)
            
            # Sistema de análisis
            system_info = f"Sistema: YOLO + {'LSTM' if lstm_model else 'Reglas'} | Device: {config.DEVICE}"
            cv2.putText(frame_display, system_info, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
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
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Reporte final
        print("\n" + "="*70)
        print("📊 REPORTE FINAL DE SESIÓN")
        print("="*70)
        print(f"Frames procesados: {frame_count}")
        print(f"Duración: {frame_count/fps:.1f} segundos")
        print(f"Total de alertas: {len(alert_history)}")
        
        if alert_history:
            print("\n📈 Distribución de alertas:")
            from collections import Counter
            
            # Por comportamiento
            behavior_counts = Counter([a['behavior'] for a in alert_history])
            print("\n  Por comportamiento:")
            for behavior, count in behavior_counts.most_common():
                print(f"    • {behavior:25s}: {count:3d} alertas")
            
            # Por nivel
            level_counts = Counter([a['level'] for a in alert_history])
            print("\n  Por nivel de riesgo:")
            level_names = {0: 'Sin riesgo', 1: 'Bajo', 2: 'Medio', 3: 'Alto'}
            for level in [3, 2, 1, 0]:
                if level in level_counts:
                    emoji = ['⚪', '🟡', '🟠', '🔴'][level]
                    print(f"    {emoji} {level_names[level]:12s}: {level_counts[level]:3d} alertas")
            
            # Por fuente
            source_counts = Counter([a['source'] for a in alert_history])
            print("\n  Por fuente de detección:")
            for source, count in source_counts.most_common():
                print(f"    • {source:20s}: {count:3d} alertas")
            
            # Confianza promedio
            avg_confidence = np.mean([a['confidence'] for a in alert_history])
            print(f"\n  Confianza promedio: {avg_confidence:.2%}")
        
        print("\n" + "="*70)
        print("✅ Sistema finalizado correctamente")
        print("="*70)


# ============================================
# PUNTO DE ENTRADA
# ============================================
if __name__ == "__main__":
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
        default=0.4,
        help='Umbral de confianza para comportamientos'
    )
    
    parser.add_argument(
        '--confidence-weapon',
        type=float,
        default=0.25,
        help='Umbral de confianza para armas'
    )
    
    args = parser.parse_args()
    
    # Actualizar configuración con argumentos
    config.CONFIDENCE_GENERAL = args.confidence_general
    config.CONFIDENCE_SOSPECHOSO = args.confidence_behavior
    config.CONFIDENCE_ARMAS = args.confidence_weapon
    config.MODELO_LSTM = args.lstm_model
    
    import multiprocessing
    multiprocessing.freeze_support()
    
    main(args)