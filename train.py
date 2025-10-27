"""
YOLO ULTIMATE TRAINING - Entrenamiento por Fases con RAM Optimizada
Autor: Sistema de entrenamiento optimizado
Ubicación: NeuroEvol-Threat-master/train-ultimate.py
"""

from ultralytics import YOLO
import torch
import os
import psutil
import time
from datetime import datetime
from pathlib import Path
import platform

# Evitar suspensión en Windows
if platform.system() == 'Windows':
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        print("✅ Protección anti-suspensión activada (Windows)")
    except:
        print("⚠️  No se pudo activar protección anti-suspensión")

# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "datasetSospecha" / "dataset.yaml"
RUNS_PATH = PROJECT_ROOT / "runs" / "detect"
LOGS_PATH = PROJECT_ROOT / "resultados"

# Crear carpetas si no existen
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def configurar_sistema():
    """Optimiza el sistema para máximo rendimiento"""
    print("\n" + "="*70)
    print("🔧 CONFIGURANDO SISTEMA PARA MÁXIMO RENDIMIENTO")
    print("="*70)
    
    # PyTorch threads
    torch.set_num_threads(os.cpu_count())
    
    # Info del sistema
    ram = psutil.virtual_memory()
    print(f"💾 RAM Total: {ram.total / (1024**3):.1f} GB")
    print(f"💾 RAM Disponible: {ram.available / (1024**3):.1f} GB ({100 - ram.percent:.1f}%)")
    print(f"🔥 CPU Cores: {os.cpu_count()}")
    print(f"🧵 PyTorch Threads: {torch.get_num_threads()}")
    print(f"📁 Dataset: {DATASET_PATH}")
    print("="*70 + "\n")

def mostrar_progreso_fase(fase, inicio):
    """Muestra información de la fase completada"""
    tiempo = (time.time() - inicio) / 60  # en minutos
    ram = psutil.virtual_memory()
    print("\n" + "="*70)
    print(f"✅ FASE {fase} COMPLETADA")
    print(f"⏱️  Tiempo: {tiempo:.1f} minutos ({tiempo/60:.1f} horas)")
    print(f"💾 RAM usada: {ram.percent:.1f}%")
    print("="*70 + "\n")

def guardar_log(contenido, nombre="entrenamiento_log.txt"):
    """Guarda log del entrenamiento"""
    log_file = LOGS_PATH / nombre
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(contenido)
    print(f"📝 Log guardado en: {log_file}")

# ==========================================
# FASE 1: ENTRENAMIENTO TURBO
# ==========================================

def fase_1_turbo():
    """
    FASE 1: Entrenamiento ultra-rápido
    - Imágenes: 320px
    - Epochs: 12
    - Tiempo estimado: 3-5 horas
    """
    print("🚀 INICIANDO FASE 1: ENTRENAMIENTO TURBO")
    print("-" * 70)
    print("📐 Tamaño: 320px | 🔄 Epochs: 12 | ⚡ Velocidad: MÁXIMA")
    print("-" * 70 + "\n")
    
    inicio = time.time()
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=str(DATASET_PATH),
        
        # Configuración ultra-rápida
        epochs=12,
        imgsz=320,
        batch=10,
        device='cpu',
        
        # Máxima RAM
        cache=True,
        workers=8,
        
        # Augmentations mínimas
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,
        degrees=3,
        translate=0.05,
        scale=0.2,
        
        # Optimizaciones
        patience=5,
        close_mosaic=3,
        optimizer='AdamW',
        
        # Guardado
        project=str(RUNS_PATH),
        name="fase1_turbo",
        exist_ok=True,
        save=True,
        save_period=4,
        verbose=True,
        plots=False,
        val=True,
    )
    
    mostrar_progreso_fase(1, inicio)
    modelo_path = RUNS_PATH / "fase1_turbo" / "weights" / "best.pt"
    
    # Guardar log
    log_content = f"Fase 1 completada\nTiempo: {(time.time() - inicio) / 60:.1f} min\nModelo: {modelo_path}\n"
    guardar_log(log_content)
    
    return str(modelo_path)

# ==========================================
# FASE 2: REFINAMIENTO
# ==========================================

def fase_2_refinado(model_anterior):
    """
    FASE 2: Refinamiento
    - Imágenes: 416px
    - Epochs: 15
    - Tiempo estimado: 5-7 horas
    """
    print("🎯 INICIANDO FASE 2: REFINAMIENTO")
    print("-" * 70)
    print("📐 Tamaño: 416px | 🔄 Epochs: 15 | ⚡ Velocidad: ALTA")
    print("-" * 70 + "\n")
    
    inicio = time.time()
    model = YOLO(model_anterior)
    
    model.train(
        data=str(DATASET_PATH),
        
        # Configuración balanceada
        epochs=15,
        imgsz=416,
        batch=8,
        device='cpu',
        
        # Máxima RAM
        cache=True,
        workers=8,
        
        # Augmentations moderadas
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=5,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        
        # Optimizaciones
        patience=8,
        close_mosaic=5,
        optimizer='AdamW',
        
        # Guardado
        project=str(RUNS_PATH),
        name="fase2_refinado",
        exist_ok=True,
        save=True,
        save_period=5,
        verbose=True,
        plots=False,
        val=True,
    )
    
    mostrar_progreso_fase(2, inicio)
    modelo_path = RUNS_PATH / "fase2_refinado" / "weights" / "best.pt"
    
    # Guardar log
    log_content = f"Fase 2 completada\nTiempo: {(time.time() - inicio) / 60:.1f} min\nModelo: {modelo_path}\n"
    guardar_log(log_content)
    
    return str(modelo_path)

# ==========================================
# FASE 3: MÁXIMA CALIDAD (OPCIONAL)
# ==========================================

def fase_3_maxima_calidad(model_anterior):
    """
    FASE 3: Máxima calidad (OPCIONAL)
    - Imágenes: 640px
    - Epochs: 10
    - Tiempo estimado: 8-12 horas
    """
    print("⭐ INICIANDO FASE 3: MÁXIMA CALIDAD")
    print("-" * 70)
    print("📐 Tamaño: 640px | 🔄 Epochs: 10 | ⚡ Velocidad: MEDIA")
    print("-" * 70 + "\n")
    
    inicio = time.time()
    model = YOLO(model_anterior)
    
    model.train(
        data=str(DATASET_PATH),
        
        # Configuración máxima calidad
        epochs=10,
        imgsz=640,
        batch=4,
        device='cpu',
        
        # RAM optimizada
        cache=True,
        workers=6,
        
        # Augmentations completas
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.0,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        
        # Optimizaciones finales
        patience=10,
        close_mosaic=3,
        optimizer='AdamW',
        
        # Guardado
        project=str(RUNS_PATH),
        name="fase3_maxima",
        exist_ok=True,
        save=True,
        save_period=3,
        verbose=True,
        plots=True,
        val=True,
    )
    
    mostrar_progreso_fase(3, inicio)
    modelo_path = RUNS_PATH / "fase3_maxima" / "weights" / "best.pt"
    
    # Guardar log
    log_content = f"Fase 3 completada\nTiempo: {(time.time() - inicio) / 60:.1f} min\nModelo: {modelo_path}\n"
    guardar_log(log_content)
    
    return str(modelo_path)

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    """Ejecuta el entrenamiento completo por fases"""
    
    print("\n" + "🎓 "*25)
    print("YOLO ULTIMATE TRAINING - FASES + RAM OPTIMIZADA")
    print("🎓 "*25 + "\n")
    
    inicio_total = time.time()
    
    # Verificar que existe el dataset
    if not DATASET_PATH.exists():
        print(f"❌ ERROR: No se encuentra el dataset en: {DATASET_PATH}")
        print("   Verifica la ruta y vuelve a intentar.")
        return
    
    # Configurar sistema
    configurar_sistema()
    
    print("📋 PLAN DE ENTRENAMIENTO AUTOMÁTICO:")
    print("   • Fase 1: Turbo (320px, 12 epochs) ~ 3-5 horas")
    print("   • Fase 2: Refinado (416px, 15 epochs) ~ 5-7 horas")
    print("   • Fase 3: Máxima (640px, 10 epochs) ~ 8-12 horas")
    print("⏱️  TIEMPO TOTAL ESTIMADO: 16-24 horas\n")
    
    respuesta = input("⏸️  ¿Iniciar entrenamiento completo (3 fases)? (s/n): ").lower()
    if respuesta != 's':
        print("❌ Entrenamiento cancelado")
        return
    
    print("\n")
    
    # ============== FASE 1 ==============
    try:
        modelo_fase1 = fase_1_turbo()
        print(f"💾 Modelo Fase 1: {modelo_fase1}\n")
    except Exception as e:
        print(f"❌ Error en Fase 1: {e}")
        return
    
    # ============== FASE 2 ==============
    try:
        modelo_fase2 = fase_2_refinado(modelo_fase1)
        print(f"💾 Modelo Fase 2: {modelo_fase2}\n")
    except Exception as e:
        print(f"❌ Error en Fase 2: {e}")
        return
    
    # ============== FASE 3 (AUTOMÁTICA) ==============
    # Ejecuta Fase 3 automáticamente
    modelo_fase3 = None
    
    try:
        modelo_fase3 = fase_3_maxima_calidad(modelo_fase2)
        print(f"💾 Modelo Fase 3: {modelo_fase3}\n")
    except Exception as e:
        print(f"❌ Error en Fase 3: {e}")
        print("⚠️  Continuando con modelo de Fase 2...")
    
    # ============== RESUMEN FINAL ==============
    tiempo_total = (time.time() - inicio_total) / 3600
    
    print("\n" + "🎉 "*30)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("🎉 "*30)
    print(f"\n⏱️  Tiempo total: {tiempo_total:.2f} horas")
    print(f"💾 RAM final: {psutil.virtual_memory().percent:.1f}%")
    print(f"\n📁 MODELOS GENERADOS:")
    print(f"   • Fase 1 (Turbo):    {modelo_fase1}")
    print(f"   • Fase 2 (Refinado): {modelo_fase2}")
    if modelo_fase3:
        print(f"   • Fase 3 (Máxima):   {modelo_fase3}")
        print(f"\n💡 RECOMENDACIÓN: Usa el modelo de Fase 3 (máxima calidad)")
    else:
        print(f"\n💡 RECOMENDACIÓN: Usa el modelo de Fase 2")
    print(f"📁 Logs guardados en: {LOGS_PATH}\n")
    
    # Guardar resumen final
    resumen = f"""
RESUMEN DEL ENTRENAMIENTO
Tiempo total: {tiempo_total:.2f} horas
Modelo Fase 1: {modelo_fase1}
Modelo Fase 2: {modelo_fase2}
Modelo Fase 3: {modelo_fase3 if modelo_fase3 else 'No ejecutado'}
"""
    guardar_log(resumen, "resumen_final.txt")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    main()