"""
YOLOv8 con Algoritmos Genéticos - SISTEMA COMPLETO
✅ Optimización evolutiva
✅ Visualización en tiempo real
✅ Checkpoints para pausar/reanudar
✅ Comparación con baseline
"""

from ultralytics import YOLO
import multiprocessing
import random
import numpy as np
from deap import base, creator, tools, algorithms
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# ============================================
# CONFIGURACIÓN BASE
# ============================================
DATASET_PATH = "datasetArmas/dataset.yaml"
BASE_MODEL = "yolov8n.pt"
DEVICE = 0  # GPU RTX 2050

HARDWARE_CONFIG = {
    'workers': 2,
    'cache': False,
    'batch': 4,
}

# ============================================
# PESOS DE FITNESS MULTI-OBJETIVO
# ============================================
W1_F1_SCORE = 10.0
W2_FP_RATE = 15.0
W3_LATENCY = 1.0

# ============================================
# RANGOS DE HIPERPARÁMETROS
# ============================================
HYPERPARAMS_RANGES = {
    'lr0': (0.0001, 0.01),
    'batch': (4, 16),
    'conf': (0.1, 0.5),
    'iou': (0.3, 0.7),
}

# ============================================
# CONFIGURACIÓN DEL GA
# ============================================
GA_CONFIG = {
    'population_size': 6,
    'generations': 3,
    'cx_prob': 0.7,
    'mut_prob': 0.2,
    'tournsize': 3,
    'validation_epochs': 5,
}

FINAL_TRAINING_EPOCHS = 50

# ============================================
# DIRECTORIOS
# ============================================
LOG_DIR = "ga_logs"
CHECKPOINT_DIR = "ga_checkpoints"
PLOTS_DIR = "ga_plots"

for directory in [LOG_DIR, CHECKPOINT_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# SISTEMA DE CHECKPOINTS
# ============================================
class CheckpointManager:
    """Maneja guardado y carga de checkpoints del GA"""
    
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'ga_checkpoint.pkl')
    
    def save(self, generation, population, hof, logbook):
        """Guarda el estado actual del GA (sin stats que contiene lambda)"""
        checkpoint_data = {
            'generation': generation,
            'population': population,
            'hof': hof,
            'logbook': logbook,
            'timestamp': datetime.now().isoformat(),
            'config': GA_CONFIG
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"💾 Checkpoint guardado: generación {generation}")
    
    def load(self):
        """Carga un checkpoint previo"""
        if not os.path.exists(self.checkpoint_file):
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n✅ Checkpoint cargado: generación {data['generation']}")
            print(f"   Timestamp: {data['timestamp']}")
            return data
        except Exception as e:
            print(f"⚠️  Error al cargar checkpoint: {e}")
            return None
    
    def exists(self):
        """Verifica si existe un checkpoint"""
        return os.path.exists(self.checkpoint_file)


# ============================================
# SISTEMA DE VISUALIZACIÓN
# ============================================
class GAVisualizer:
    """Genera gráficas de la evolución del GA"""
    
    def __init__(self, plots_dir=PLOTS_DIR):
        self.plots_dir = plots_dir
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_evolution(self, logbook, filename='evolution.png'):
        """Gráfica de evolución del fitness"""
        gen = logbook.select("gen")
        avg_fitness = logbook.select("avg")
        max_fitness = logbook.select("max")
        min_fitness = logbook.select("min")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(gen, max_fitness, 'g-', label='Mejor Fitness', linewidth=2)
        ax.plot(gen, avg_fitness, 'b--', label='Fitness Promedio', linewidth=2)
        ax.plot(gen, min_fitness, 'r:', label='Peor Fitness', linewidth=2)
        
        ax.fill_between(gen, min_fitness, max_fitness, alpha=0.2, color='blue')
        
        ax.set_xlabel('Generación', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Evolución del Fitness por Generación', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráfica guardada: {save_path}")
    
    def plot_hyperparams_distribution(self, hof, filename='hyperparams_dist.png'):
        """Distribución de hiperparámetros en los mejores individuos"""
        if len(hof) == 0:
            return
        
        params = {
            'lr0': [ind[0] for ind in hof],
            'batch': [int(ind[1]) for ind in hof],
            'conf': [ind[2] for ind in hof],
            'iou': [ind[3] for ind in hof]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Distribución de Hiperparámetros (Top 3)', 
                     fontsize=14, fontweight='bold')
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for idx, (param_name, values) in enumerate(params.items()):
            ax = axes[idx // 2, idx % 2]
            ax.bar(range(len(values)), values, color=colors[:len(values)], alpha=0.7)
            ax.set_title(param_name.upper(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Ranking', fontsize=10)
            ax.set_ylabel('Valor', fontsize=10)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([f'#{i+1}' for i in range(len(values))])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráfica guardada: {save_path}")
    
    def plot_comparison(self, baseline_results, ga_results, filename='comparison.png'):
        """Compara resultados baseline vs GA optimizado"""
        metrics = ['F1-Score', 'Precision', 'Recall', 'FP Rate', 'Latency (ms)']
        
        baseline_vals = [
            baseline_results.get('f1', 0),
            baseline_results.get('precision', 0),
            baseline_results.get('recall', 0),
            baseline_results.get('fp_rate', 0),
            baseline_results.get('latency', 0) * 1000
        ]
        
        ga_vals = [
            ga_results.get('f1', 0),
            ga_results.get('precision', 0),
            ga_results.get('recall', 0),
            ga_results.get('fp_rate', 0),
            ga_results.get('latency', 0) * 1000
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
                       color='#95a5a6', alpha=0.8)
        bars2 = ax.bar(x + width/2, ga_vals, width, label='GA Optimizado', 
                       color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
        ax.set_title('Comparación: Baseline vs GA Optimizado', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores sobre las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráfica de comparación guardada: {save_path}")


# ============================================
# LOGGING
# ============================================
current_generation = 0
individual_counter = 0

def log_individual(generation, individual_id, individual, fitness, metrics):
    """Guarda registro de cada individuo"""
    log_file = os.path.join(LOG_DIR, f"generation_{generation}.jsonl")
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'generation': generation,
        'individual_id': individual_id,
        'hyperparams': {
            'lr0': individual[0],
            'batch': int(individual[1]),
            'conf': individual[2],
            'iou': individual[3],
        },
        'fitness': fitness,
        'metrics': metrics
    }
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============================================
# CONFIGURACIÓN DE DEAP
# ============================================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_lr0", random.uniform, 
                 HYPERPARAMS_RANGES['lr0'][0], 
                 HYPERPARAMS_RANGES['lr0'][1])
toolbox.register("attr_batch", random.choice, [4, 8, 12, 16])
toolbox.register("attr_conf", random.uniform, 
                 HYPERPARAMS_RANGES['conf'][0], 
                 HYPERPARAMS_RANGES['conf'][1])
toolbox.register("attr_iou", random.uniform, 
                 HYPERPARAMS_RANGES['iou'][0], 
                 HYPERPARAMS_RANGES['iou'][1])

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lr0, toolbox.attr_batch, 
                  toolbox.attr_conf, toolbox.attr_iou), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ============================================
# FUNCIÓN DE FITNESS
# ============================================
def evaluate_individual(individual: list) -> tuple:
    """Evalúa un individuo con fitness multi-objetivo"""
    global individual_counter
    individual_counter += 1
    
    params = {
        'lr0': individual[0],
        'batch': int(individual[1]),
        'conf': individual[2],
        'iou': individual[3],
        'epochs': GA_CONFIG['validation_epochs'],
        'data': DATASET_PATH,
        'device': DEVICE,
        'workers': HARDWARE_CONFIG['workers'],
        'cache': HARDWARE_CONFIG['cache'],
        'verbose': False,
        'patience': 0,
        'name': f'ga_gen{current_generation}_ind{individual_counter}'
    }
    
    try:
        print(f"\n🔬 Evaluando individuo {individual_counter}:")
        print(f"   LR={individual[0]:.6f}, Batch={int(individual[1])}, "
              f"Conf={individual[2]:.3f}, IoU={individual[3]:.3f}")
        
        start_time = time.time()
        
        model = YOLO(BASE_MODEL)
        training_results = model.train(
            lr0=params['lr0'],
            batch=params['batch'],
            epochs=params['epochs'],
            data=params['data'],
            device=params['device'],
            workers=params['workers'],
            cache=params['cache'],
            verbose=params['verbose'],
            patience=params['patience'],
            name=params['name']
        )
        
        results = model.val(
            data=params['data'],
            conf=params['conf'],
            iou=params['iou'],
            device=params['device'],
            verbose=False
        )
        
        eval_time = time.time() - start_time
        
        # ✅ CORRECCIÓN: Usar .mean() para obtener promedio de todas las clases
        f1_score = float(results.box.f1.mean()) if results.box.f1 is not None else 0.0
        precision = float(results.box.p.mean()) if results.box.p is not None else 0.0
        recall = float(results.box.r.mean()) if results.box.r is not None else 0.0
        latency = results.speed['inference'] / 1000.0
        fp_rate = max(0.0, 1.0 - precision)
        
        fitness_value = (W1_F1_SCORE * f1_score) - \
                       (W2_FP_RATE * fp_rate) - \
                       (W3_LATENCY * latency)
        
        metrics = {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'fp_rate': fp_rate,
            'latency': latency,
            'eval_time': eval_time
        }
        
        log_individual(current_generation, individual_counter, individual, 
                      fitness_value, metrics)
        
        print(f"   ✓ F1={f1_score:.4f}, Precision={precision:.4f}, "
              f"FP_Rate={fp_rate:.4f}, Latency={latency:.4f}s")
        print(f"   → Fitness={fitness_value:.4f} (evaluado en {eval_time:.1f}s)")
        
        return fitness_value,
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        log_individual(current_generation, individual_counter, individual,
                      -1000.0, {'error': str(e)})
        return -1000.0,


def custom_mutate(individual, indpb=0.2):
    """Mutación personalizada"""
    if random.random() < indpb:
        individual[0] = random.uniform(*HYPERPARAMS_RANGES['lr0'])
    if random.random() < indpb:
        individual[1] = random.choice([4, 8, 12, 16])
    if random.random() < indpb:
        individual[2] = random.uniform(*HYPERPARAMS_RANGES['conf'])
    if random.random() < indpb:
        individual[3] = random.uniform(*HYPERPARAMS_RANGES['iou'])
    return individual,


toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=GA_CONFIG['tournsize'])


# ============================================
# BASELINE (ENTRENAMIENTO ESTÁNDAR)
# ============================================
def train_baseline():
    """Entrena modelo baseline sin optimización"""
    print("\n" + "=" * 80)
    print("📊 ENTRENAMIENTO BASELINE (sin optimización)")
    print("=" * 80)
    
    model = YOLO(BASE_MODEL)
    
    print("\nEntrenando con configuración estándar...")
    results = model.train(
        data=DATASET_PATH,
        epochs=GA_CONFIG['validation_epochs'],
        batch=8,
        device=DEVICE,
        workers=HARDWARE_CONFIG['workers'],
        cache=HARDWARE_CONFIG['cache'],
        name='baseline',
        verbose=False
    )
    
    val_results = model.val(data=DATASET_PATH)
    
    # ✅ CORRECCIÓN: Usar .mean() para todas las métricas
    baseline_metrics = {
        'f1': float(val_results.box.f1.mean()) if val_results.box.f1 is not None else 0.0,
        'precision': float(val_results.box.p.mean()) if val_results.box.p is not None else 0.0,
        'recall': float(val_results.box.r.mean()) if val_results.box.r is not None else 0.0,
        'fp_rate': max(0.0, 1.0 - float(val_results.box.p.mean())) if val_results.box.p is not None else 1.0,
        'latency': val_results.speed['inference'] / 1000.0
    }
    
    print("\n✅ Baseline completado")
    print(f"   F1-Score: {baseline_metrics['f1']:.4f}")
    print(f"   Precision: {baseline_metrics['precision']:.4f}")
    print(f"   FP Rate: {baseline_metrics['fp_rate']:.4f}")
    
    # Guardar baseline
    baseline_file = os.path.join(LOG_DIR, 'baseline_results.json')
    with open(baseline_file, 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
    
    return baseline_metrics


# ============================================
# ALGORITMO GENÉTICO PRINCIPAL
# ============================================
def run_genetic_algorithm(resume=False):
    """Ejecuta el GA con soporte para checkpoints"""
    global current_generation, individual_counter
    
    checkpoint_mgr = CheckpointManager()
    visualizer = GAVisualizer()
    
    print("=" * 80)
    print("🧬 ALGORITMO GENÉTICO PARA OPTIMIZACIÓN DE YOLOv8")
    print("=" * 80)
    
    # Verificar dataset
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ ERROR: No se encontró {DATASET_PATH}")
        return None, None
    
    # Intentar reanudar desde checkpoint
    if resume and checkpoint_mgr.exists():
        print("\n🔄 Reanudando desde checkpoint...")
        checkpoint = checkpoint_mgr.load()
        if checkpoint:
            population = checkpoint['population']
            hof = checkpoint['hof']
            logbook = checkpoint['logbook']
            start_gen = checkpoint['generation'] + 1
            current_generation = start_gen
        else:
            print("⚠️  No se pudo cargar checkpoint, iniciando desde cero")
            resume = False
    
    if not resume:
        print(f"\n📋 Configuración:")
        print(f"   Dataset: {DATASET_PATH}")
        print(f"   Población: {GA_CONFIG['population_size']}")
        print(f"   Generaciones: {GA_CONFIG['generations']}")
        print(f"   Épocas de validación: {GA_CONFIG['validation_epochs']}")
        
        population = toolbox.population(n=GA_CONFIG['population_size'])
        hof = tools.HallOfFame(3)
        logbook = tools.Logbook()
        start_gen = 0
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print(f"\n🚀 Iniciando evolución desde generación {start_gen}...\n")
    
    # Evolución con checkpoints
    for gen in range(start_gen, GA_CONFIG['generations']):
        current_generation = gen
        individual_counter = 0
        
        print(f"\n{'='*80}")
        print(f"GENERACIÓN {gen + 1}/{GA_CONFIG['generations']}")
        print(f"{'='*80}")
        
        # Evaluar población
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Actualizar Hall of Fame
        hof.update(population)
        
        # Registrar estadísticas
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        
        print(f"\n📊 Estadísticas Generación {gen}:")
        print(f"   Mejor: {record['max']:.4f}")
        print(f"   Promedio: {record['avg']:.4f}")
        print(f"   Peor: {record['min']:.4f}")
        
        # ✅ CORRECCIÓN: No pasar stats al checkpoint
        checkpoint_mgr.save(gen, population, hof, logbook)
        
        # Generar gráficas
        if gen > 0:
            visualizer.plot_evolution(logbook, f'evolution_gen{gen}.png')
        
        # Siguiente generación
        if gen < GA_CONFIG['generations'] - 1:
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < GA_CONFIG['cx_prob']:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < GA_CONFIG['mut_prob']:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            population[:] = offspring
    
    # Gráficas finales
    visualizer.plot_evolution(logbook, 'evolution_final.png')
    visualizer.plot_hyperparams_distribution(hof, 'hyperparams_final.png')
    
    # Mostrar mejores individuos
    print("\n" + "=" * 80)
    print("🏆 MEJORES INDIVIDUOS")
    print("=" * 80)
    for i, ind in enumerate(hof, 1):
        print(f"\n#{i} - Fitness: {ind.fitness.values[0]:.4f}")
        print(f"   LR: {ind[0]:.6f}, Batch: {int(ind[1])}, "
              f"Conf: {ind[2]:.4f}, IoU: {ind[3]:.4f}")
    
    # Guardar mejores individuos
    best_file = os.path.join(LOG_DIR, 'best_individuals.json')
    with open(best_file, 'w') as f:
        json.dump([{
            'rank': i,
            'fitness': ind.fitness.values[0],
            'hyperparams': {
                'lr0': ind[0],
                'batch': int(ind[1]),
                'conf': ind[2],
                'iou': ind[3]
            }
        } for i, ind in enumerate(hof, 1)], f, indent=2)
    
    return hof, logbook


# ============================================
# ENTRENAMIENTO FINAL
# ============================================
def train_best_model(best_individual, epochs=None):
    """Entrena modelo final con mejores hiperparámetros"""
    if epochs is None:
        epochs = FINAL_TRAINING_EPOCHS
    
    print("\n" + "=" * 80)
    print("🎯 ENTRENAMIENTO FINAL")
    print("=" * 80)
    
    params = {
        'lr0': best_individual[0],
        'batch': int(best_individual[1]),
        'conf': best_individual[2],
        'iou': best_individual[3],
        'epochs': epochs,
        'data': DATASET_PATH,
        'device': DEVICE,
        'workers': HARDWARE_CONFIG['workers'],
        'cache': HARDWARE_CONFIG['cache'],
        'name': 'best_model_ga_optimized',
        'verbose': True,
    }
    
    print(f"\n📊 Hiperparámetros optimizados:")
    for k, v in params.items():
        if k not in ['data', 'device', 'workers', 'cache', 'verbose']:
            print(f"   {k}: {v}")
    
    model = YOLO(BASE_MODEL)
    results = model.train(**params)
    
    val_results = model.val(data=params['data'])
    
    # ✅ CORRECCIÓN: Usar .mean() para todas las métricas
    ga_metrics = {
        'f1': float(val_results.box.f1.mean()) if val_results.box.f1 is not None else 0.0,
        'precision': float(val_results.box.p.mean()) if val_results.box.p is not None else 0.0,
        'recall': float(val_results.box.r.mean()) if val_results.box.r is not None else 0.0,
        'fp_rate': max(0.0, 1.0 - float(val_results.box.p.mean())) if val_results.box.p is not None else 1.0,
        'latency': val_results.speed['inference'] / 1000.0
    }
    
    print("\n✅ Entrenamiento completado")
    print(f"   Modelo: runs/detect/{params['name']}/weights/best.pt")
    
    return model, results, ga_metrics


# ============================================
# COMPARACIÓN
# ============================================
def compare_results():
    """Compara baseline vs GA optimizado"""
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN: BASELINE VS GA OPTIMIZADO")
    print("=" * 80)
    
    # Cargar resultados
    baseline_file = os.path.join(LOG_DIR, 'baseline_results.json')
    ga_file = os.path.join(LOG_DIR, 'ga_optimized_results.json')
    
    if not os.path.exists(baseline_file):
        print("\n⚠️  No se encontró baseline. Ejecuta train_baseline() primero")
        return
    
    if not os.path.exists(ga_file):
        print("\n⚠️  No se encontró GA optimizado.")