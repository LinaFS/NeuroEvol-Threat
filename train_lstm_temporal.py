"""
train_lstm_temporal.py - Entrenamiento del modelo LSTM para comportamientos
VERSIÓN CORREGIDA - Compatible con PyTorch 1.x y 2.x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Importar configuración de clases
import sys
sys.path.append('.')
from classes_config import BEHAVIOR_CLASSES


# ============================================
# CONFIGURACIÓN
# ============================================
class TrainConfig:
    # Datos
    DATA_DIR = 'data/temporal_annotations'
    TRAIN_FILE = 'train_annotations.json'
    VAL_FILE = 'val_annotations.json'
    
    # Modelo
    INPUT_DIM = 20
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NUM_CLASSES = 6
    DROPOUT = 0.3
    
    # Entrenamiento
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0  # Cambiado a 0 para evitar problemas en Windows
    
    # Output
    MODEL_DIR = 'models'
    CHECKPOINT_DIR = 'checkpoints'
    PLOTS_DIR = 'plots'

config = TrainConfig()


# ============================================
# MODELO LSTM
# ============================================
class BehaviorLSTM(nn.Module):
    """
    Modelo LSTM para clasificación de comportamientos temporales
    """
    def __init__(self, input_dim=20, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super(BehaviorLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Capas de clasificación
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Normalización
        self.batch_norm = nn.BatchNorm1d(hidden_dim // 2)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, num_classes)
            hidden: hidden state
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Tomar último output
        last_output = lstm_out[:, -1, :]
        
        # Clasificación
        out = self.fc1(last_output)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, hidden


# ============================================
# MAPEO DE CLASES
# ============================================
# Mapeo de 21 clases de comportamiento a 6 categorías principales
BEHAVIOR_MAPPING = {
    # Normal (0)
    'Normal_Videos': 0,
    'Walking': 0,
    'Clapping': 0,
    'Walking_While_Reading_Book': 0,
    'Walking_While_Using_Phone': 0,
    'Sitting': 0,
    'Standing_Still': 0,
    
    # Loitering (1)
    'Meet_and_Split': 1,
    
    # Aggression (2)
    'Abuse': 2,
    'Assault': 2,
    'Fighting': 2,
    
    # Weapon Carry (3)
    'Shooting': 3,
    
    # Erratic Movement (4)
    'Burglary': 4,
    'Stealing': 4,
    'Shoplifting': 4,
    'Vandalism': 4,
    'Arrest': 4,
    
    # Critical (5)
    'Arson': 5,
    'Explosion': 5,
    'Robbery': 5,
    'RoadAccidents': 5,
}

# Clases principales para el LSTM
LSTM_CLASSES = {
    0: 'normal',
    1: 'loitering',
    2: 'aggression',
    3: 'weapon_carry',
    4: 'erratic_movement',
    5: 'critical'
}


# ============================================
# GENERADOR DE DATOS SINTÉTICOS
# ============================================
class SyntheticDataGenerator:
    """
    Genera datos sintéticos para entrenar el modelo
    """
    
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.window_size = 30
    
    def generate(self):
        """Generar dataset sintético"""
        samples = []
        
        print("   Generando muestras sintéticas...")
        
        for i in range(self.num_samples):
            # Mostrar progreso
            if (i + 1) % 200 == 0:
                print(f"      {i+1}/{self.num_samples} muestras generadas")
            
            # Seleccionar comportamiento aleatorio
            behavior_id = np.random.randint(0, 6)
            behavior = LSTM_CLASSES[behavior_id]
            
            # Generar secuencia basada en comportamiento
            sequence = self._generate_behavior_sequence(behavior)
            
            samples.append({
                'features': sequence,
                'label': behavior_id,
                'behavior': behavior
            })
        
        print(f"      {self.num_samples}/{self.num_samples} muestras generadas")
        
        return samples
    
    def _generate_behavior_sequence(self, behavior):
        """Generar secuencia de características para un comportamiento"""
        sequence = []
        
        if behavior == 'normal':
            # Movimiento lineal constante
            for i in range(self.window_size):
                features = np.array([
                    0.5 + i*0.01,  # x_mean (movimiento lineal)
                    0.5,           # y_mean
                    0.02,          # x_std (baja variación)
                    0.02,          # y_std
                    0.01,          # area_coverage
                    0.0,           # near_edge
                    2.0,           # velocity_mean (constante)
                    3.0,           # velocity_max
                    0.5,           # velocity_std (baja)
                    0.1,           # acceleration_mean
                    0.5,           # acceleration_max
                    1.0,           # direction_changes (pocos)
                    1.0,           # dwelling_time
                    50.0,          # distance_traveled
                    1.0,           # trajectory_duration
                    30.0,          # frames_count
                    0.0,           # nearby_objects
                    100.0,         # min_distance
                    0.0,           # interaction_duration
                    0.0            # zone_visited
                ], dtype=np.float32)
                sequence.append(features)
        
        elif behavior == 'loitering':
            # Permanencia prolongada, poco movimiento
            base_x, base_y = 0.5, 0.5
            for i in range(self.window_size):
                features = np.array([
                    base_x + np.random.randn()*0.01,  # Casi estático
                    base_y + np.random.randn()*0.01,
                    0.05,  # Mayor dispersión espacial
                    0.05,
                    0.02,
                    0.0,
                    0.5,   # Velocidad muy baja
                    1.0,
                    0.3,
                    0.05,
                    0.1,
                    2.0,   # Pocos cambios de dirección
                    10.5,  # Tiempo prolongado
                    10.0,  # Poca distancia
                    10.5,
                    30.0,
                    0.0,
                    100.0,
                    0.0,
                    1.0
                ], dtype=np.float32)
                sequence.append(features)
        
        elif behavior == 'aggression':
            # Movimiento rápido y errático
            for i in range(self.window_size):
                features = np.array([
                    0.5 + np.random.randn()*0.1,
                    0.5 + np.random.randn()*0.1,
                    0.15,  # Alta variación
                    0.15,
                    0.05,
                    0.0,
                    12.0,  # Velocidad alta
                    15.0,
                    4.0,   # Alta desviación
                    8.0,   # Aceleración alta
                    12.0,
                    5.0,   # Muchos cambios
                    2.0,
                    150.0,
                    2.0,
                    30.0,
                    2.0,   # Objetos cercanos
                    30.0,
                    1.0,
                    2.0
                ], dtype=np.float32)
                sequence.append(features)
        
        elif behavior == 'weapon_carry':
            # Similar a normal pero con arma
            for i in range(self.window_size):
                features = np.array([
                    0.5 + i*0.01,
                    0.5,
                    0.03,
                    0.03,
                    0.01,
                    0.0,
                    2.5,
                    4.0,
                    0.8,
                    0.3,
                    0.8,
                    2.0,
                    2.0,
                    60.0,
                    2.0,
                    30.0,
                    0.0,
                    100.0,
                    0.0,
                    1.0
                ], dtype=np.float32)
                sequence.append(features)
        
        elif behavior == 'erratic_movement':
            # Muchos cambios de dirección
            for i in range(self.window_size):
                features = np.array([
                    0.5 + np.sin(i*0.5)*0.1,  # Zigzag
                    0.5 + np.cos(i*0.5)*0.1,
                    0.12,
                    0.12,
                    0.04,
                    0.0,
                    5.0,
                    8.0,
                    3.0,
                    3.0,
                    5.0,
                    8.0,   # Muchos cambios de dirección
                    3.0,
                    100.0,
                    3.0,
                    30.0,
                    1.0,
                    50.0,
                    0.5,
                    2.0
                ], dtype=np.float32)
                sequence.append(features)
        
        else:  # critical
            # Eventos críticos (explosión, incendio, etc.)
            for i in range(self.window_size):
                features = np.array([
                    0.5 + np.random.randn()*0.2,
                    0.5 + np.random.randn()*0.2,
                    0.20,  # Muy alta variación
                    0.20,
                    0.08,
                    0.0,
                    15.0,  # Velocidad muy alta
                    20.0,
                    5.0,
                    10.0,
                    15.0,
                    8.0,
                    1.5,
                    200.0,
                    1.5,
                    30.0,
                    3.0,
                    20.0,
                    0.8,
                    3.0
                ], dtype=np.float32)
                sequence.append(features)
        
        return np.array(sequence, dtype=np.float32)


# ============================================
# DATASET
# ============================================
class TemporalBehaviorDataset(Dataset):
    """Dataset para secuencias temporales"""
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = torch.FloatTensor(sample['features'])
        label = torch.LongTensor([sample['label']])
        
        return features, label


# ============================================
# ENTRENAMIENTO
# ============================================
class Trainer:
    """Clase para entrenar el modelo"""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # CORRECCIÓN: Remover 'verbose' que no existe en versiones antiguas
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=5,
            factor=0.5
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """Entrenar una época"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for features, labels in pbar:
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE).squeeze()
            
            # Forward
            self.optimizer.zero_grad()
            outputs, _ = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Actualizar barra
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validar modelo"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc='Validation'):
                features = features.to(config.DEVICE)
                labels = labels.to(config.DEVICE).squeeze()
                
                outputs, _ = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, epochs):
        """Ciclo completo de entrenamiento"""
        print(f"\n🚀 Iniciando entrenamiento ({epochs} épocas)")
        print(f"   Device: {config.DEVICE}")
        print(f"   Train samples: {len(self.train_loader.dataset)}")
        print(f"   Val samples: {len(self.val_loader.dataset)}")
        print("="*70)
        
        for epoch in range(epochs):
            print(f"\n📍 Época {epoch+1}/{epochs}")
            
            # Entrenar
            train_loss, train_acc = self.train_epoch()
            
            # Validar
            val_loss, val_acc, preds, labels = self.validate()
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Scheduler
            self.scheduler.step(val_acc)
            
            # Resumen
            print(f"\n   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best')
                print(f"   ✅ Mejor modelo guardado (Val Acc: {val_acc:.2f}%)")
            
            # Checkpoint cada 10 épocas
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}')
        
        print("\n" + "="*70)
        print(f"✅ Entrenamiento completado!")
        print(f"   Mejor Val Acc: {self.best_val_acc:.2f}%")
        print("="*70)
        
        return self.history
    
    def save_checkpoint(self, epoch, name):
        """Guardar checkpoint"""
        Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': {
                'input_dim': config.INPUT_DIM,
                'hidden_dim': config.HIDDEN_DIM,
                'num_layers': config.NUM_LAYERS,
                'num_classes': config.NUM_CLASSES,
                'dropout': config.DROPOUT
            }
        }
        
        path = Path(config.CHECKPOINT_DIR) / f'{name}.pth'
        torch.save(checkpoint, path)


# ============================================
# EVALUACIÓN
# ============================================
class Evaluator:
    """Evaluación completa del modelo"""
    
    def __init__(self, model, val_loader):
        self.model = model.to(config.DEVICE)
        self.val_loader = val_loader
    
    def evaluate(self):
        """Evaluación completa"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n📊 Evaluando modelo...")
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader):
                features = features.to(config.DEVICE)
                labels = labels.to(config.DEVICE).squeeze()
                
                outputs, _ = self.model(features)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Reporte de clasificación
        print("\n" + "="*70)
        print("📈 REPORTE DE CLASIFICACIÓN")
        print("="*70)
        
        report = classification_report(
            all_labels,
            all_preds,
            target_names=list(LSTM_CLASSES.values()),
            digits=3
        )
        print(report)
        
        # Matriz de confusión
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # Estadísticas por clase
        self.plot_per_class_stats(all_labels, all_preds)
        
        return all_preds, all_labels, all_probs
    
    def plot_confusion_matrix(self, labels, preds):
        """Plotear matriz de confusión"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(LSTM_CLASSES.values()),
            yticklabels=list(LSTM_CLASSES.values()),
            cbar_kws={'label': 'Count'}
        )
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix - Behavior Classification', fontsize=14)
        plt.tight_layout()
        
        Path(config.PLOTS_DIR).mkdir(exist_ok=True)
        plt.savefig(Path(config.PLOTS_DIR) / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"\n💾 Matriz de confusión guardada en: {config.PLOTS_DIR}/confusion_matrix.png")
    
    def plot_per_class_stats(self, labels, preds):
        """Estadísticas por clase"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None
        )
        
        behaviors = list(LSTM_CLASSES.values())
        x = np.arange(len(behaviors))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Behavior', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(config.PLOTS_DIR) / 'per_class_metrics.png', dpi=300)
        plt.close()
        
        print(f"💾 Métricas por clase guardadas en: {config.PLOTS_DIR}/per_class_metrics.png")


# ============================================
# VISUALIZACIÓN
# ============================================
def plot_training_history(history):
    """Plotear historial de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(config.PLOTS_DIR).mkdir(exist_ok=True)
    plt.savefig(Path(config.PLOTS_DIR) / 'training_history.png', dpi=300)
    plt.close()
    
    print(f"💾 Historial guardado en: {config.PLOTS_DIR}/training_history.png")


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
def main():
    """Función principal"""
    
    print("="*70)
    print("🧠 ENTRENAMIENTO DE MODELO LSTM TEMPORAL")
    print("   NeuroEvol-Threat - Behavioral Analysis")
    print("="*70)
    
    # Crear directorios
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(config.PLOTS_DIR).mkdir(exist_ok=True)
    
    # Generar datos sintéticos
    print("\n📦 Generando dataset sintético...")
    generator = SyntheticDataGenerator(num_samples=2000)
    all_samples = generator.generate()
    
    # Split train/val
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"   ✅ Train: {len(train_samples)} muestras")
    print(f"   ✅ Val: {len(val_samples)} muestras")
    
    # Datasets y DataLoaders
    train_dataset = TemporalBehaviorDataset(train_samples)
    val_dataset = TemporalBehaviorDataset(val_samples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    # Modelo
    print("\n🏗️  Construyendo modelo LSTM...")
    model = BehaviorLSTM(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parámetros: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    print(f"   Device: {config.DEVICE}")
    
    # Entrenar
    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train(config.EPOCHS)
    
    # Plotear historial
    plot_training_history(history)
    
    # Guardar modelo final
    final_model_path = Path(config.MODEL_DIR) / 'behavior_lstm_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': config.INPUT_DIM,
            'hidden_dim': config.HIDDEN_DIM,
            'num_layers': config.NUM_LAYERS,
            'num_classes': config.NUM_CLASSES,
            'dropout': config.DROPOUT
        },
        'lstm_classes': LSTM_CLASSES,
        'behavior_mapping': BEHAVIOR_MAPPING
    }, final_model_path)
    print(f"\n💾 Modelo final guardado en: {final_model_path}")
    
    # Evaluación final
    evaluator = Evaluator(model, val_loader)
    evaluator.evaluate()
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)
    print("\n📁 Archivos generados:")
    print(f"  • Modelo final: {final_model_path}")
    print(f"  • Checkpoints: {config.CHECKPOINT_DIR}/")
    print(f"  • Gráficas: {config.PLOTS_DIR}/")
    print("\n💡 Para usar el modelo entrenado:")
    print(f"  python main_temporal.py --lstm-model {final_model_path}")
    print("="*70)

if __name__ == "__main__":
    main()