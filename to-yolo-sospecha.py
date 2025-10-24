import os
import csv
import cv2
from shutil import copy2
from tqdm import tqdm
from time import time
import random

csv_path = 'resultados/anotaciones_corregido.csv'
images_src = 'dataSospecha'
dataset_path = 'datasetSospecha'

# CONFIGURACIÓN DE DIVISIÓN
SPLIT_RATIO = 0.8  # 80% train, 20% val
RANDOM_SEED = 42

# Crear directorios para train y val
images_train = os.path.join(dataset_path, 'images/train')
labels_train = os.path.join(dataset_path, 'labels/train')
images_val = os.path.join(dataset_path, 'images/val')
labels_val = os.path.join(dataset_path, 'labels/val')

for directory in [images_train, labels_train, images_val, labels_val]:
    os.makedirs(directory, exist_ok=True)

annotations = {}
errores = []

print("Leyendo archivo CSV y extrayendo clases...")

# Leer CSV
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extraer clases únicas
clases_unicas = sorted(set(row['class'] for row in rows if row.get('class')))
class_map = {clase: idx for idx, clase in enumerate(clases_unicas)}

print(f"\nClases detectadas ({len(class_map)}):")
for clase, idx in class_map.items():
    print(f"  {idx}: {clase}")

print(f"\nTotal de anotaciones: {len(rows)}")

# Crear índice de carpetas disponibles
print("\nIndexando carpetas disponibles...")
carpetas_disponibles = {}
for item in os.listdir(images_src):
    item_path = os.path.join(images_src, item)
    if os.path.isdir(item_path):
        carpetas_disponibles[item] = item_path

print(f"✓ Se encontraron {len(carpetas_disponibles)} carpetas")

imagenes_no_encontradas = set()

for row in tqdm(rows, desc="Procesando anotaciones CSV"):
    try:
        start = time()
        csv_filename = row['image_filename']
        cls = row['class']
        
        class_id = class_map.get(cls)
        if class_id is None:
            print(f"[!] Clase desconocida '{cls}' en imagen {csv_filename}")
            errores.append(f"{csv_filename} (clase desconocida: {cls})")
            continue

        nombre_base = csv_filename.rsplit('_', 1)[0]
        
        if nombre_base not in carpetas_disponibles:
            if csv_filename not in imagenes_no_encontradas:
                imagenes_no_encontradas.add(csv_filename)
            continue
        
        carpeta_path = carpetas_disponibles[nombre_base]
        
        imagenes_en_carpeta = sorted([
            f for f in os.listdir(carpeta_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        if not imagenes_en_carpeta:
            if csv_filename not in imagenes_no_encontradas:
                imagenes_no_encontradas.add(csv_filename)
            continue
        
        try:
            indice_str = csv_filename.rsplit('_', 1)[1].split('.')[0]
            indice = int(indice_str)
            
            if indice >= len(imagenes_en_carpeta):
                if csv_filename not in imagenes_no_encontradas:
                    imagenes_no_encontradas.add(csv_filename)
                continue
            
            nombre_real = imagenes_en_carpeta[indice]
            img_path = os.path.join(carpeta_path, nombre_real)
            
        except (ValueError, IndexError):
            if csv_filename not in imagenes_no_encontradas:
                imagenes_no_encontradas.add(csv_filename)
            continue

        x_min = int(row['x_min'])
        y_min = int(row['y_min'])
        x_max = int(row['x_max'])
        y_max = int(row['y_max'])

        img = cv2.imread(img_path)

        if img is None:
            print(f"[!] No se pudo leer la imagen: {img_path}")
            errores.append(csv_filename)
            continue

        h, w = img.shape[:2]

        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_w = (x_max - x_min) / w
        bbox_h = (y_max - y_min) / h

        if csv_filename not in annotations:
            annotations[csv_filename] = {'path': img_path, 'boxes': []}

        annotations[csv_filename]['boxes'].append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        )

        if time() - start > 2:
            print(f"[!] Imagen lenta: {csv_filename}")

    except Exception as e:
        print(f"[ERROR] {row.get('image_filename', 'desconocida')} → {e}")
        errores.append(row.get('image_filename', 'desconocida'))
        continue

# ============================================
# DIVIDIR EN TRAIN Y VAL
# ============================================
print("\n📊 Dividiendo dataset en train/val...")

# Obtener lista de todas las imágenes únicas
all_images = list(annotations.keys())
random.seed(RANDOM_SEED)
random.shuffle(all_images)

# Calcular punto de división
split_idx = int(len(all_images) * SPLIT_RATIO)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

print(f"\n📊 División del dataset:")
print(f"   Total: {len(all_images)} imágenes")
print(f"   Train: {len(train_images)} imágenes ({SPLIT_RATIO*100:.0f}%)")
print(f"   Val: {len(val_images)} imágenes ({(1-SPLIT_RATIO)*100:.0f}%)")

# ============================================
# GUARDAR ARCHIVOS
# ============================================
print("\nGuardando archivos YOLO y copiando imágenes...")

def save_annotations(image_list, images_dir, labels_dir, split_name):
    """Guarda las anotaciones para un split específico"""
    for csv_filename in tqdm(image_list, desc=f"Guardando {split_name}"):
        try:
            data = annotations[csv_filename]
            src_img = data['path']
            dst_img = os.path.join(images_dir, csv_filename)
            
            # Copiar imagen
            if not os.path.exists(dst_img):
                copy2(src_img, dst_img)

            # Guardar label
            label_file = os.path.splitext(csv_filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'w', encoding='utf-8') as f:
                for annot in data['boxes']:
                    f.write(annot + '\n')

        except Exception as e:
            print(f"[ERROR guardando {csv_filename}]: {e}")
            errores.append(csv_filename)

# Guardar train
save_annotations(train_images, images_train, labels_train, "train")

# Guardar val
save_annotations(val_images, images_val, labels_val, "val")

# ============================================
# GUARDAR CLASES
# ============================================
classes_file = os.path.join(dataset_path, 'classes.txt')
with open(classes_file, 'w', encoding='utf-8') as f:
    for clase in clases_unicas:
        f.write(f"{clase}\n")

# ============================================
# RESUMEN FINAL
# ============================================
print(f"\n✅ ¡Dataset listo para entrenamiento!")
print(f"\n📁 Estructura creada:")
print(f"   {dataset_path}/")
print(f"   ├── images/")
print(f"   │   ├── train/ ({len(train_images)} imágenes)")
print(f"   │   └── val/ ({len(val_images)} imágenes)")
print(f"   ├── labels/")
print(f"   │   ├── train/ ({len(train_images)} archivos)")
print(f"   │   └── val/ ({len(val_images)} archivos)")
print(f"   └── classes.txt")

print(f"\n📝 Archivo de clases: {classes_file}")

if imagenes_no_encontradas:
    print(f"\n⚠️  Imágenes no encontradas: {len(imagenes_no_encontradas)}")
    print("Primeras 10:")
    for img in list(imagenes_no_encontradas)[:10]:
        print(f" - {img}")
    if len(imagenes_no_encontradas) > 10:
        print(f"   ... y {len(imagenes_no_encontradas) - 10} más")

if errores:
    print(f"\n⚠️  Otros errores: {len(errores)}")
    for err in errores[:10]:
        print(f" - {err}")
    if len(errores) > 10:
        print(f"   ... y {len(errores) - 10} más")