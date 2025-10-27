# NeuroEvol-Threat 🧬🔍

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![DEAP](https://img.shields.io/badge/DEAP-Evolutionary-green.svg)](https://github.com/DEAP/deap)

> Sistema avanzado de detección de amenazas en tiempo real mediante **optimización evolutiva**, **análisis de patrones temporales** y **aprendizaje profundo** para reconocimiento de comportamientos sospechosos.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características Principales](#-características-principales)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Problema a Resolver](#-problema-a-resolver)
- [Objetivos](#-objetivos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Pipeline de Detección](#-pipeline-de-detección)
- [Reconocimiento de Patrones](#-reconocimiento-de-patrones)
- [Optimización Evolutiva](#-optimización-evolutiva)
- [Dataset](#-dataset)
- [Resultados](#-resultados)
- [Roadmap](#-roadmap)
- [Contribución](#-contribución)

---

## 🎯 Descripción

**NeuroEvol-Threat** es un sistema inteligente de videovigilancia que combina tres tecnologías clave:

1. **Detección de Objetos (YOLOv8)**: Identificación precisa de personas, armas y objetos sospechosos
2. **Optimización Evolutiva (Algoritmos Genéticos)**: Ajuste automático de hiperparámetros para maximizar precisión y minimizar falsos positivos
3. **Reconocimiento de Patrones Temporales**: Análisis de secuencias de comportamiento para detectar actividades anómalas

### ¿Qué hace diferente a este sistema?

A diferencia de los sistemas tradicionales que solo detectan **objetos estáticos**, NeuroEvol-Threat analiza **patrones de comportamiento temporal** como:

- 🚶 **Loitering**: Permanencia prolongada en zonas restringidas
- 🤜 **Agresión**: Movimientos violentos o forcejeos entre personas
- 🎒 **Objetos abandonados**: Mochilas, bolsas o paquetes sin supervisión
- 🔫 **Portación de armas**: Detección de armas de fuego, cuchillos u objetos contundentes
- 🏃 **Movimientos erráticos**: Trayectorias inusuales o cambios bruscos de dirección
- 👥 **Agrupaciones anómalas**: Formación de grupos en áreas no autorizadas

---

## ✨ Características Principales

### 🧬 Optimización Evolutiva
- **Algoritmos Genéticos (GA)** para búsqueda automática de hiperparámetros óptimos
- Función de fitness **multi-objetivo** que balancea:
  - F1-Score (peso 10x)
  - Tasa de Falsos Positivos (peso 15x - prioridad)
  - Latencia de inferencia (peso 1x)
- Sistema de **checkpoints** para pausar/reanudar entrenamiento
- **Visualización en tiempo real** de la evolución

### 🎯 Detección de Objetos
- **YOLOv8** optimizado para:
  - Detección de personas en escenarios complejos
  - Identificación de armas (pistolas, cuchillos, rifles)
  - Objetos sospechosos (mochilas, paquetes, vehículos)
- Procesamiento con **aceleración GPU** (CUDA/OpenCL)
- Detección robusta en condiciones de baja iluminación

### 📊 Reconocimiento de Patrones Temporales
- **Tracking multi-objeto** con DeepSORT/ByteTrack
- **Análisis de trayectorias** espaciales de objetos/personas
- **Ventana temporal deslizante** para captura de secuencias
- **Clasificación de comportamientos** mediante:
  - LSTM para secuencias cortas
  - Transformers para contexto largo
- Detección de anomalías basada en:
  - Velocidad de movimiento
  - Permanencia en áreas específicas
  - Proximidad entre objetos
  - Cambios bruscos de dirección

### 🚀 Sistema de Producción
- Procesamiento en **tiempo real** (30+ FPS)
- **Multi-modelo** (detección general + detección de armas)
- Sistema de **alertas graduales** (bajo, medio, alto riesgo)
- **Dashboard web** para monitoreo (opcional)
- Logs estructurados en JSON para análisis posterior

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Video Stream                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PASO 1: Detección de Objetos                    │
│         ┌───────────────────────────────────┐                │
│         │   YOLOv8 Multi-Modelo             │                │
│         │   - Modelo General (personas)     │                │
│         │   - Modelo Armas (optimizado GA)  │                │
│         └───────────────────────────────────┘                │
│                       │                                       │
│                       ├─► Personas detectadas                │
│                       ├─► Armas detectadas                   │
│                       └─► Objetos sospechosos                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         PASO 2: Tracking & Extracción de Features           │
│         ┌───────────────────────────────────┐                │
│         │   DeepSORT / ByteTrack            │                │
│         │   - Asignación de IDs únicos      │                │
│         │   - Tracking multi-objeto         │                │
│         └───────────────────────────────────┘                │
│                       │                                       │
│                       ├─► Trayectorias (x, y, t)             │
│                       ├─► Velocidades                        │
│                       ├─► Áreas de interés                   │
│                       └─► Historial de detecciones           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│       PASO 3: Análisis de Patrones Temporales               │
│         ┌───────────────────────────────────┐                │
│         │   Módulo de Secuencias            │                │
│         │   - Ventana deslizante (30 frames)│                │
│         │   - Extracción de características │                │
│         │     · Velocidad media/máxima      │                │
│         │     · Tiempo de permanencia       │                │
│         │     · Cambios de dirección        │                │
│         │     · Distancia recorrida         │                │
│         └───────────────────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│         ┌───────────────────────────────────┐                │
│         │   LSTM / Transformer              │                │
│         │   - Input: Secuencia de features  │                │
│         │   - Output: Probabilidad anomalía │                │
│         └───────────────────────────────────┘                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            PASO 4: Clasificación de Amenazas                 │
│         ┌───────────────────────────────────┐                │
│         │   Reglas de Negocio               │                │
│         │   + Modelo de Clasificación       │                │
│         └───────────────────────────────────┘                │
│                       │                                       │
│                       ├─► Loitering (permanencia)            │
│                       ├─► Agresión (forcejeo)                │
│                       ├─► Objeto abandonado                  │
│                       ├─► Portación de arma                  │
│                       └─► Movimiento errático                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT: Alertas                            │
│         ┌───────────────────────────────────┐                │
│         │   Sistema de Alertas Graduales    │                │
│         │   - 🟢 Bajo:   Monitoreo rutinario│                │
│         │   - 🟡 Medio:  Atención requerida │                │
│         │   - 🔴 Alto:   Acción inmediata   │                │
│         └───────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────┐
        │   Capa de Optimización Evolutiva     │
        │   (Algoritmos Genéticos - GA)        │
        │   - Optimización de hiperparámetros  │
        │   - Reducción de falsos positivos    │
        │   - Balance precisión/velocidad      │
        └──────────────────────────────────────┘
```

---

## ⚠️ Problema a Resolver

### 1. Limitación en Detección de Patrones Complejos
Los sistemas tradicionales:
- ❌ Detectan objetos de manera **estática** (personas, mochilas, armas)
- ❌ **No analizan comportamiento** ni secuencias temporales
- ❌ Generan **exceso de alertas** sin contexto temporal
- ❌ Incapaces de distinguir entre comportamiento **normal y sospechoso**

**Ejemplo**: Detectar una persona con mochila ≠ Detectar una persona que deja su mochila y se aleja (objeto abandonado)

### 2. Optimización Ineficiente
- ❌ Ajuste manual de hiperparámetros (lento, subóptimo)
- ❌ Grid search/random search son **costosos** computacionalmente
- ❌ Altos **falsos positivos** en producción
- ❌ Difícil balance entre **precisión y velocidad**

---

## 🎯 Objetivos

### Objetivo General
Desarrollar un sistema de detección de amenazas que combine **detección de objetos**, **análisis temporal de comportamiento** y **optimización evolutiva** para lograr alta precisión con mínimos falsos positivos en tiempo real.

### Objetivos Específicos

1. ✅ **Implementar detección de objetos** con YOLOv8 optimizado
   - Modelo para personas (general)
   - Modelo para armas (especializado)

2. 🔄 **Desarrollar módulo de tracking** multi-objeto
   - DeepSORT/ByteTrack para asignación de IDs
   - Historial de trayectorias espaciales

3. 🔄 **Implementar análisis de patrones temporales**
   - Extracción de características de secuencias
   - Modelo LSTM/Transformer para clasificación
   - Detección de anomalías comportamentales

4. ✅ **Optimizar con algoritmos evolutivos**
   - Algoritmos Genéticos para hiperparámetros
   - Función multi-objetivo (F1, FP Rate, Latency)
   - Reducción de 14-33% en falsos positivos

5. 🔄 **Evaluar con métricas robustas**
   - Precisión, Recall, F1-Score
   - Tasa de Falsos Positivos/Hora
   - FPS (frames por segundo)
   - AUC-ROC para clasificación de anomalías

6. 🔄 **Comparar con métodos baseline**
   - YOLOv8 sin optimización
   - Sistemas sin análisis temporal
   - Métodos tradicionales de optimización

---

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- GPU NVIDIA (recomendado) con CUDA 11.8+
- 8GB+ RAM (16GB recomendado)
- pip o conda

### Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/LinaFS/NeuroEvol-Threat.git
cd NeuroEvol-Threat

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación GPU (opcional)
python codigos-grafica/try-grafica.py
```

### Dependencias Principales
```
# Core
ultralytics>=8.0.0        # YOLOv8
torch>=2.0.0              # PyTorch (con CUDA)
opencv-python>=4.8.0      # Procesamiento de video

# Optimización Evolutiva
deap>=1.4.0               # Algoritmos Genéticos
numpy>=1.24.0
scipy>=1.10.0

# Tracking & Análisis Temporal
filterpy>=1.4.5           # Filtros de Kalman (para tracking)
scikit-learn>=1.3.0       # Machine Learning

# Visualización & Logging
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0

# GPU Optimization
numba>=0.57.0             # Aceleración JIT
```

---

## 🚀 Uso

### 1. Entrenamiento con Optimización Evolutiva

```bash
# Entrenamiento completo (baseline → GA → final)
python train-GA.py

# Ver estado del pipeline
python train-GA.py status

# Reanudar desde checkpoint
python train-GA.py resume

# Reiniciar desde cero
python train-GA.py restart
```

**Configuración rápida** (2-8 horas):
```python
GA_CONFIG = {
    'population_size': 4,
    'generations': 2,
    'validation_epochs': 3,
}
FINAL_TRAINING_EPOCHS = 30
```

### 2. Detección en Tiempo Real

```bash
# Webcam
python main.py --source 0

# Archivo de video
python main.py --source video.mp4

# Stream RTSP
python main.py --source rtsp://usuario:pass@ip:puerto/stream
```

### 3. Procesamiento de Dataset

```bash
# Extraer frames de videos
python preprocesamiento/frame.py

# Generar anotaciones automáticas
python preprocesamiento/trainning.py           # Para actividades sospechosas
python preprocesamiento/trainning-grafica-weapons.py  # Para armas

# Corregir clases y convertir a formato YOLO
python preprocesamiento/corregir_csv.py
python preprocesamiento/to-yolo.py
```

---

## 🔄 Pipeline de Detección

### Fase 1: Preprocesamiento
```python
# Extracción de frames (1 cada 10 frames)
python preprocesamiento/frame.py
# Output: dataSospecha/Video001_frames/frame_00000.jpg
```

### Fase 2: Anotación Automática
```python
# Detección multi-método (HOG + Haar Cascade + Segmentación)
python preprocesamiento/trainning.py
# Output: resultados/anotaciones.csv
```

### Fase 3: Entrenamiento Base
```python
# Conversión a formato YOLO
python preprocesamiento/to-yolo.py
# Output: datasetSospecha/
#   ├── images/train/
#   ├── images/val/
#   ├── labels/train/
#   └── labels/val/
```

### Fase 4: Optimización Evolutiva
```python
# Algoritmo Genético
python train-GA.py
# Logs: ga_logs/
# Gráficas: ga_plots/
```

**Resultados obtenidos** (hardware: RTX 2050):
- ✅ **F1-Score**: +9.67% (0.6725 → 0.7376)
- ✅ **Precision**: +6.09% (0.7042 → 0.7471)
- ✅ **Recall**: +13.54% (0.6462 → 0.7337)
- ✅ **FP Rate**: -14.50% (0.2958 → 0.2529)

---

## 🧠 Reconocimiento de Patrones

### Arquitectura del Módulo Temporal

```python
# Extracción de características de trayectorias
class TemporalFeatureExtractor:
    def extract_features(self, trajectory, window_size=30):
        """
        trajectory: List[(x, y, timestamp, class_id, confidence)]
        """
        features = {
            'velocity_mean': self.compute_velocity(trajectory),
            'velocity_std': self.compute_velocity_std(trajectory),
            'direction_changes': self.count_direction_changes(trajectory),
            'dwelling_time': self.compute_dwelling_time(trajectory),
            'distance_traveled': self.compute_distance(trajectory),
            'area_occupancy': self.compute_area_occupancy(trajectory),
            'trajectory_curvature': self.compute_curvature(trajectory)
        }
        return features

# Modelo de clasificación temporal
class BehaviorClassifier(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        # Tomar última salida
        output = self.fc(lstm_out[:, -1, :])
        return F.softmax(output, dim=1)
```

### Patrones Detectados

| Patrón | Características | Umbral |
|--------|----------------|--------|
| **Loitering** | `velocity_mean < 0.5 px/frame` y `dwelling_time > 5s` | Medio/Alto |
| **Agresión** | `velocity_std > 2.0` y `direction_changes > 5` en ventana | Alto |
| **Objeto Abandonado** | `object_static > 10s` y `person_distance > 50px` | Alto |
| **Portación Arma** | `weapon_detected` + `person_nearby` | Alto |
| **Movimiento Errático** | `curvature > 0.8` y `direction_changes > 7` | Medio |

---

## 🧬 Optimización Evolutiva

### Función de Fitness Multi-Objetivo

```python
Fitness = (10 × F1-Score) - (15 × FP_Rate) - (1 × Latency)
```

**Explicación de pesos**:
- **F1-Score (10x)**: Balance entre precisión y recall
- **FP Rate (15x)**: **Prioridad máxima** - minimizar falsas alarmas
- **Latency (1x)**: Consideración de velocidad sin sacrificar precisión

### Hiperparámetros Optimizados

| Hiperparámetro | Rango | Óptimo (RTX 2050) |
|---------------|-------|-------------------|
| `lr0` | 0.0001 - 0.01 | 0.004398 |
| `batch` | 4, 8, 12, 16 | 8 |
| `conf` | 0.1 - 0.5 | 0.4866 |
| `iou` | 0.3 - 0.7 | 0.4908 |

### Proceso Evolutivo

```
Generación 0: 6 individuos aleatorios
             ↓ Evaluación (5 épocas c/u)
             ↓ Selección por torneo
Generación 1: 6 individuos (cruce + mutación)
             ↓ Evaluación
             ↓ Selección
Generación 2: 6 individuos refinados
             ↓ Evaluación
             ↓ Hall of Fame (Top 3)
Entrenamiento Final: Mejor individuo (50 épocas)
```

---

## 📊 Dataset

### Fuentes de Datos

1. **UCF-Crime** (1,900 videos, 13 clases)
   - Eventos delictivos en videos largos
   - Anotaciones temporales débiles

2. **XD-Violence** (4,754 videos)
   - Dataset multimodal (audio + video)
   - 6 categorías de violencia

3. **Dataset Propio** (en construcción)
   - Escenarios escolares/urbanos
   - Anotaciones precisas frame-level

### Clases de Actividades

| Categoría | Subcategorías | Ejemplos |
|-----------|--------------|----------|
| **Intrusión** | Acceso no autorizado | Saltar bardas, forzar puertas |
| **Agresión** | Violencia física | Peleas, forcejeos, empujones |
| **Abandono** | Objetos sin supervisión | Mochilas, paquetes > 10s |
| **Armas** | Objetos peligrosos | Pistolas, cuchillos, bates |
| **Errático** | Movimiento anómalo | Merodeo, zigzag, cambios bruscos |
| **Normal** | Comportamiento estándar | Caminar, conversar, esperar |

### Estructura del Dataset

```
datasetSospecha/
├── images/
│   ├── train/          (80% - 2,400 imágenes)
│   └── val/            (20% - 600 imágenes)
├── labels/
│   ├── train/          (formato YOLO: class x_center y_center width height)
│   └── val/
├── classes.txt         (lista de clases)
└── dataset.yaml        (configuración para YOLOv8)
```

---

## 📈 Resultados

### Comparación Baseline vs GA Optimizado

| Métrica | Baseline | GA Optimizado | Mejora |
|---------|----------|---------------|--------|
| F1-Score | 0.6725 | 0.7376 | ✅ **+9.67%** |
| Precision | 0.7042 | 0.7471 | ✅ +6.09% |
| Recall | 0.6462 | 0.7337 | ✅ **+13.54%** |
| FP Rate | 0.2958 | 0.2529 | ✅ **-14.50%** |
| Latency | 4.8ms | 4.9ms | ⚠️ -1.01% |

### Visualizaciones Generadas

1. **evolution_final.png**: Evolución del fitness por generación
2. **hyperparams_final.png**: Distribución de hiperparámetros óptimos
3. **comparison_final.png**: Comparación baseline vs optimizado

### Hardware Utilizado

- **GPU**: NVIDIA RTX 2050 (4GB VRAM)
- **CPU**: Intel Core (workers=2)
- **Tiempo total**: ~7.8 horas (baseline + GA + final)

---

## 🗺️ Roadmap

### ✅ Fase 1: Fundamentos (Completada)
- [x] Configuración del proyecto
- [x] Implementación de YOLOv8 base
- [x] Preprocesamiento y anotación automática
- [x] Conversión a formato YOLO
- [x] Entrenamiento baseline

### ✅ Fase 2: Optimización Evolutiva (Completada)
- [x] Implementación de Algoritmos Genéticos
- [x] Función multi-objetivo (F1, FP Rate, Latency)
- [x] Sistema de checkpoints
- [x] Visualización de evolución
- [x] Reducción de 14.5% en falsos positivos

### 🔄 Fase 3: Reconocimiento de Patrones (En Progreso)
- [ ] Integración de DeepSORT/ByteTrack
- [ ] Extracción de características temporales
- [ ] Modelo LSTM para clasificación de comportamiento
- [ ] Detección de anomalías (loitering, agresión, etc.)
- [ ] Sistema de alertas graduales

### 📅 Fase 4: Evaluación y Despliegue
- [ ] Pruebas en escenarios reales
- [ ] Métricas de detección temporal (AUC-ROC)
- [ ] Dashboard web de monitoreo
- [ ] Optimización para Jetson Nano/Edge devices
- [ ] Documentación completa

### 📅 Fase 5: Mejoras Avanzadas
- [ ] Transformer para contexto largo (> 30 frames)
- [ ] Multi-cámara con fusión de información
- [ ] Detección de audio (gritos, disparos)
- [ ] Re-identificación de personas entre cámaras
- [ ] Predicción de trayectorias futuras

---

## 🤝 Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama: `git checkout -b feature/PatternRecognition`
3. Commit: `git commit -m 'Add LSTM temporal classifier'`
4. Push: `git push origin feature/PatternRecognition`
5. Abre un Pull Request

### Áreas que necesitan contribución:
- 🔴 **Alta prioridad**: Módulo de tracking multi-objeto
- 🔴 **Alta prioridad**: Modelo LSTM/Transformer
- 🟡 **Media prioridad**: Dashboard web
- 🟢 **Baja prioridad**: Integración con sistemas de alarma

---

## 📚 Referencias

### Papers
- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **DeepSORT**: Simple Online and Realtime Tracking with a Deep Association Metric
- **UCF-Crime**: Real-world Anomaly Detection in Surveillance Videos
- **XD-Violence**: Not only Look, but also Listen: Cross-modal Dataset

### Datasets
- [UCF-Crime Dataset](https://www.crcv.ucf.edu/projects/real-world/)
- [XD-Violence Dataset](https://roc-ng.github.io/XD-Violence/)

### Tools
- [DEAP - Distributed Evolutionary Algorithms](https://github.com/DEAP/deap)
- [DeepSORT PyTorch](https://github.com/ZQPei/deep_sort_pytorch)

---

## 👥 Autores

- **LinaFS** - Desarrollo principal - [GitHub](https://github.com/LinaFS)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- Ultralytics por YOLOv8
- Comunidad de DEAP por el framework evolutivo
- Autores de UCF-Crime y XD-Violence datasets
- Comunidad de Computer Vision

---

## 📧 Contacto

Para preguntas, colaboraciones o reportar issues:
- **GitHub Issues**: [NeuroEvol-Threat/issues](https://github.com/LinaFS/NeuroEvol-Threat/issues)
- **Email**: [Disponible en el perfil de GitHub]

---

⭐ **Si este proyecto te resulta útil, considera darle una estrella en GitHub**

🚀 **Estado del Proyecto**: En desarrollo activo (Fase 3/5)
