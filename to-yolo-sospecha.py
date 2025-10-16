import os
import csv
import cv2
from shutil import copy2
from tqdm import tqdm
from time import time

csv_path = 'resultados/anotaciones.csv'
images_src = 'dataSospecha'
dataset_path = 'proyecto/datasetSospecha'

images_train = os.path.join(dataset_path, 'images/train')
labels_train = os.path.join(dataset_path, 'labels/train')
os.makedirs(images_train, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)

class_map = {'Arma': 0}
annotations = {}

print("Leyendo archivo CSV...")

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total de anotaciones: {len(rows)}")
errores = []

for row in tqdm(rows, desc="Procesando anotaciones CSV"):
    try:
        start = time()
        img_name = row['image_filename']
        cls = row['class']
        class_id = class_map.get(cls, 0)

        x_min = int(row['x_min'])
        y_min = int(row['y_min'])
        x_max = int(row['x_max'])
        y_max = int(row['y_max'])

        img_path = os.path.join(images_src, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[!] No se pudo leer la imagen: {img_name}")
            errores.append(img_name)
            continue

        h, w = img.shape[:2]

        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_w = (x_max - x_min) / w
        bbox_h = (y_max - y_min) / h

        if img_name not in annotations:
            annotations[img_name] = []

        annotations[img_name].append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
        )

        if time() - start > 2:
            print(f"[!] Imagen lenta: {img_name}")

    except Exception as e:
        print(f"[ERROR] {row.get('image_filename', 'desconocida')} → {e}")
        errores.append(row.get('image_filename', 'desconocida'))
        continue

print("Guardando archivos YOLO y copiando imágenes...")

for img_name in tqdm(annotations, desc="Copiando imágenes y etiquetas"):
    try:
        annots = annotations[img_name]
        src_img = os.path.join(images_src, img_name)
        dst_img = os.path.join(images_train, img_name)
        if not os.path.exists(dst_img):
            copy2(src_img, dst_img)

        label_file = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_train, label_file)
        with open(label_path, 'w') as f:
            for annot in annots:
                f.write(annot + '\n')

    except Exception as e:
        print(f"[ERROR guardando {img_name}]: {e}")
        errores.append(img_name)

print("✅ ¡Dataset listo para entrenamiento!")

if errores:
    print(f"\nSe encontraron {len(errores)} errores:")
    for err in errores[:10]:  # Muestra solo los primeros 10
        print(f" - {err}")
