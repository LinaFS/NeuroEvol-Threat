# NeuroEvol-Threat 🧬🔍

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

> Real-time suspicious activity detection system powered by evolutionary algorithms and neural networks for enhanced pattern recognition and threat identification.

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Problema a Resolver](#-problema-a-resolver)
- [Objetivos](#-objetivos)
- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Dataset](#-dataset)
- [Metodología](#-metodología)
- [Roadmap](#-roadmap)
- [Contribución](#-contribución)

## 🎯 Descripción

**NeuroEvol-Threat** es un sistema inteligente de videovigilancia que combina **redes neuronales convolucionales (CNN)** con **algoritmos evolutivos** para detectar actividades sospechosas en tiempo real. A diferencia de los sistemas tradicionales que solo identifican objetos, este proyecto reconoce **patrones complejos de comportamiento** como:

- 🚶 Permanencia prolongada en zonas restringidas
- 🤜 Forcejeos o interacciones agresivas
- 🎒 Abandono de objetos por períodos extensos
- 🔫 Portación de armas u objetos peligrosos
- 🏃 Movimientos erráticos o trayectorias inusuales

## ✨ Características

- **Detección en Tiempo Real**: Procesamiento de video en vivo con YOLOv8
- **Análisis Conductual**: Reconocimiento de patrones espacio-temporales sospechosos
- **Optimización Evolutiva**: Ajuste automático de hiperparámetros mediante algoritmos genéticos
- **Reducción de Falsos Positivos**: Mayor precisión gracias a la optimización bioinspirada
- **Arquitectura Híbrida**: Combinación de detección de objetos + análisis temporal
- **Adaptabilidad**: Optimización específica para diferentes condiciones (iluminación, cámaras)

## ⚠️ Problema a Resolver

Los sistemas de videovigilancia actuales enfrentan **dos problemas críticos**:

### 1. Limitación en la Detección de Patrones Complejos
- Los modelos detectan objetos de manera **estática** (personas, mochilas, armas)
- **Carecen de análisis conductual** para reconocer comportamientos sospechosos
- Los datasets disponibles son **pequeños, poco variados** o mal anotados
- Dificultad para **generalizar** a situaciones reales y diversas

### 2. Optimización Ineficiente de Hiperparámetros
- El ajuste se realiza con **métodos tradicionales** (grid search, random search)
- Son **costosos computacionalmente** y no garantizan resultados óptimos
- Resulta en modelos con **falsos positivos/negativos** que reducen su aplicabilidad

## 🎯 Objetivos

### Objetivo General
Desarrollar e implementar un sistema inteligente de detección de situaciones sospechosas en tiempo real mediante visión artificial y aprendizaje profundo, optimizando sus hiperparámetros y desempeño con algoritmos evolutivos.

### Objetivos Específicos

1. Diseñar un **pipeline de detección** basado en YOLOv8 para identificar objetos y patrones de comportamiento
2. Construir un **dataset representativo** de situaciones de riesgo con anotaciones de calidad
3. Identificar **hiperparámetros clave** que influyen en el rendimiento del sistema
4. Implementar **algoritmos de optimización evolutiva** (GA, PSO) para ajuste automático
5. Evaluar el sistema con métricas robustas: **precisión, recall, F1-score, FPS, AUC**
6. Comparar el desempeño evolutivo vs. métodos tradicionales de ajuste

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Video Stream                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              YOLOv8 Object Detection                         │
│         (Personas, Armas, Mochilas, Objetos)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Object Tracking (DeepSORT/ByteTrack)               │
│              Seguimiento de Trayectorias                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        Temporal Pattern Analysis                             │
│   (LSTM/Transformer para análisis espacio-temporal)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Suspicious Behavior Detection                     │
│     - Permanencia prolongada                                 │
│     - Forcejeos                                              │
│     - Abandono de objetos                                    │
│     - Movimientos erráticos                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT: Alerts                             │
└─────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────┐
        │   Evolutionary Optimization Layer     │
        │   (GA/PSO para ajuste de             │
        │    hiperparámetros en tiempo real)    │
        └──────────────────────────────────────┘
```

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- CUDA (opcional, para GPU)
- pip o conda

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/LinaFS/NeuroEvol-Threat.git
cd NeuroEvol-Threat

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales
```
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
deap>=1.4.0  # Para algoritmos evolutivos
pymoo>=0.6.0  # Optimización multiobjetivo
scipy>=1.10.0
matplotlib>=3.7.0
```

## 🚀 Uso

### Detección Básica en Video

```python
from neuroevol_threat import ThreatDetector

# Inicializar el detector
detector = ThreatDetector(
    model_path='weights/yolov8n.pt',
    confidence=0.5
)

# Procesar video
detector.detect_from_video('video.mp4', output='output.mp4')
```

### Optimización Evolutiva de Hiperparámetros

```python
from neuroevol_threat import EvolutionaryOptimizer

# Configurar optimizador
optimizer = EvolutionaryOptimizer(
    algorithm='genetic',  # 'genetic', 'pso', 'differential_evolution'
    population_size=20,
    generations=50
)

# Ejecutar optimización
best_params = optimizer.optimize(
    dataset='data/train',
    metric='f1_score'
)

print(f"Mejores hiperparámetros: {best_params}")
```

### Detección en Tiempo Real (Webcam)

```python
detector.detect_realtime(source=0)  # 0 = webcam por defecto
```

## 📊 Dataset

El proyecto utiliza y expande datasets públicos:

- **UCF-Crime**: Eventos delictivos en videos largos no recortados
- **XD-Violence**: Dataset multimodal (audio+video) con etiquetas débiles
- **UCA (UCF-Crime Action)**: Anotaciones lingüísticas finas
- **Dataset Propio**: Escenarios específicos de entornos escolares

### Categorías de Actividades Sospechosas

| Categoría | Descripción | Ejemplos |
|-----------|-------------|----------|
| **Intrusión** | Acceso no autorizado a zonas restringidas | Persona saltando bardas, forzando puertas |
| **Agresión** | Conductas violentas o forcejeos | Peleas, empujones, amenazas físicas |
| **Abandono de objetos** | Objetos dejados por períodos prolongados | Mochilas, bolsas, paquetes sin supervisión |
| **Portación de armas** | Objetos peligrosos visibles | Armas de fuego, navajas, objetos contundentes |
| **Comportamiento errático** | Movimientos inusuales o sospechosos | Merodeo, cambios bruscos de dirección |

## 🧬 Metodología

### 1. Pipeline de Detección

```
Video Frame → Preprocesamiento → YOLOv8 → Tracking → Análisis Temporal → Clasificación
```

### 2. Optimización Evolutiva

**Hiperparámetros a Optimizar:**
- Learning rate
- Batch size
- Confidence threshold
- NMS threshold
- Data augmentation parameters
- Arquitectura de capas temporales

**Algoritmos Implementados:**
- Algoritmos Genéticos (GA)
- Particle Swarm Optimization (PSO)
- Differential Evolution (DE)

### 3. Métricas de Evaluación

- **Detección de Objetos**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Detección de Anomalías**: AUC-ROC, F1-Score, Falsos Positivos/Hora
- **Rendimiento**: FPS, Latencia, Time-to-Detect

## 🗺️ Roadmap

### Fase 1: Fundamentos ✅
- [x] Configuración del proyecto
- [x] Implementación de YOLOv8 base
- [ ] Recolección y anotación de dataset inicial

### Fase 2: Detección Avanzada 🚧
- [ ] Integración de object tracking (DeepSORT/ByteTrack)
- [ ] Módulo de análisis temporal (LSTM/Transformer)
- [ ] Detección de patrones sospechosos

### Fase 3: Optimización Evolutiva 📅
- [ ] Implementación de algoritmo genético
- [ ] Implementación de PSO
- [ ] Framework de evaluación automática

### Fase 4: Evaluación y Despliegue 📅
- [ ] Pruebas en escenarios reales
- [ ] Comparación con métodos baseline
- [ ] Documentación completa
- [ ] Sistema de alertas en tiempo real

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📚 Referencias

- Ultralytics YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- UCF-Crime Dataset: [https://www.crcv.ucf.edu/projects/real-world/](https://www.crcv.ucf.edu/projects/real-world/)
- XD-Violence Dataset: [https://roc-ng.github.io/XD-Violence/](https://roc-ng.github.io/XD-Violence/)

## 👥 Autores

- **LinaFS** - [GitHub](https://github.com/LinaFS)

## 🙏 Agradecimientos

- Ultralytics por YOLOv8
- Comunidad de Computer Vision
- Datasets públicos de investigación

---

⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub

📧 Para preguntas o colaboraciones, abre un issue en el repositorio
