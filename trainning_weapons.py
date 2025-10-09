import cv2
import os
import csv
from tqdm import tqdm

# Ruta a la carpeta principal de imágenes
data_dir = 'dataArmas'  # <- cambia según necesites

# Carpeta de salida para anotaciones
output_root = os.path.join('proyecto', 'resultados')
os.makedirs(output_root, exist_ok=True)

# Ruta del archivo CSV
annotations_path = os.path.join(output_root, 'anotaciones_Armas.csv')
csv_header = ['image_filename', 'class', 'x_min', 'y_min', 'x_max', 'y_max']
csv_rows = []

# Clase fija para las detecciones automáticas
default_class = 'arma'

# Obtener lista de imágenes
image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]

# Procesar imágenes
for filename in tqdm(image_files, desc="Procesando imágenes"):
    image_path = os.path.join(data_dir, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"No se pudo leer {image_path}")
        continue

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

        # Guardar CSV
        csv_rows.append([filename, default_class, x_min, y_min, x_max, y_max])

# Guardar CSV de respaldo
with open(annotations_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_rows)

# Resumen
print(f"\n✔ Anotaciones CSV guardadas en: {annotations_path}")
print(f"✔ Total de imágenes procesadas: {len(image_files)}")
print(f"✔ Total de detecciones registradas: {len(csv_rows)}")
