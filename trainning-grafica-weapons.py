import os
import csv
import cv2
from shutil import copy2
from tqdm import tqdm
from time import time

csv_path = 'resultados/anotaciones_corregido.csv'
images_src = 'dataSospecha'
dataset_path = 'datasetSospecha'

images_train = os.path.join(dataset_path, 'images/train')
labels_train = os.path.join(dataset_path, 'labels/train')
os.makedirs(images_train, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)

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
        csv_filename = row['image_filename']  # ej: Abuse001_x264_frames_0001.jpg
        cls = row['class']
        
        # Obtener class_id
        class_id = class_map.get(cls)
        if class_id is None:
            print(f"[!] Clase desconocida '{cls}' en imagen {csv_filename}")
            errores.append(f"{csv_filename} (clase desconocida: {cls})")
            continue

        # Extraer el nombre de la carpeta del nombre del CSV
        # Formato: "NombreCarpeta_####.ext" → extraer "NombreCarpeta"
        nombre_base = csv_filename.rsplit('_', 1)[0]  # Quita el último "_####"
        
        # Buscar la carpeta correspondiente
        if nombre_base not in carpetas_disponibles:
            if csv_filename not in imagenes_no_encontradas:
                imagenes_no_encontradas.add(csv_filename)
            continue
        
        carpeta_path = carpetas_disponibles[nombre_base]
        
        # Listar todas las imágenes en esa carpeta
        imagenes_en_carpeta = sorted([
            f for f in os.listdir(carpeta_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        if not imagenes_en_carpeta:
            if csv_filename not in imagenes_no_encontradas:
                imagenes_no_encontradas.add(csv_filename)
            continue
        
        # Extraer el índice del nombre del CSV
        # Formato: "NombreCarpeta_0001.jpg" → extraer índice 1
        try:
            indice_str = csv_filename.rsplit('_', 1)[1].split('.')[0]  # "0001"
            indice = int(indice_str)
            
            # Validar que el índice esté dentro del rango
            if indice >= len(imagenes_en_carpeta):
                if csv_filename not in imagenes_no_encontradas:
                    imagenes_no_encontradas.add(csv_filename)
                continue
            
            # Obtener el nombre real de la imagen en la carpeta
            nombre_real = imagenes_en_carpeta[indice]
            img_path = os.path.join(carpeta_path, nombre_real)
            
        except (ValueError, IndexError):
            if csv_filename not in imagenes_no_encontradas:
                imagenes_no_encontradas.add(csv_filename)
            continue

        # Procesar coordenadas
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

        # Usar el nombre del CSV como key para mantener la referencia
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

print("\nGuardando archivos YOLO y copiando imágenes...")

for csv_filename in tqdm(annotations, desc="Copiando imágenes y etiquetas"):
    try:
        data = annotations[csv_filename]
        src_img = data['path']
        dst_img = os.path.join(images_train, csv_filename)
        
        if not os.path.exists(dst_img):
            copy2(src_img, dst_img)

        label_file = os.path.splitext(csv_filename)[0] + '.txt'
        label_path = os.path.join(labels_train, label_file)
        with open(label_path, 'w', encoding='utf-8') as f:
            for annot in data['boxes']:
                f.write(annot + '\n')

    except Exception as e:
        print(f"[ERROR guardando {csv_filename}]: {e}")
        errores.append(csv_filename)

# Guardar archivo de clases
classes_file = os.path.join(dataset_path, 'classes.txt')
with open(classes_file, 'w', encoding='utf-8') as f:
    for clase in clases_unicas:
        f.write(f"{clase}\n")

print(f"\n✅ ¡Dataset listo para entrenamiento!")
print(f"📝 Archivo de clases guardado en: {classes_file}")
print(f"📊 Imágenes procesadas: {len(annotations)}")

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