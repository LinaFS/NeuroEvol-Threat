import os
import cv2
import numpy as np
from tkinter import Tk, filedialog
from tqdm import tqdm

# Ruta base donde guardarás las imágenes
BASE_DIR = r"C:\Users\paufu\OneDrive\Documentos\Python\proyecto"

def seleccionar_imagenes():
    root = Tk()
    root.withdraw()
    carpeta = filedialog.askdirectory(
        title="Selecciona la carpeta con las imágenes"
    )
    
    if not carpeta:
        return []
    
    # Buscar todas las imágenes en la carpeta
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
    imagenes = []
    
    for f in os.listdir(carpeta):
        if f.lower().endswith(extensiones):
            imagenes.append(os.path.join(carpeta, f))
    
    return sorted(imagenes)

def mejorar_calidad(img):
    """Mejora la calidad de la imagen para mejor detección"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    brightness = 30
    img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img

def segmentar_imagen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, large_contours, -1, 255, thickness=cv2.FILLED)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented, large_contours

def procesar_lote_imagenes():
    """
    Procesa imágenes y las guarda en carpetas organizadas
    """
    imagenes = seleccionar_imagenes()
    if not imagenes:
        print("No se seleccionaron imágenes.")
        return

    # Crear subcarpetas
    mejoradas_dir = os.path.join(BASE_DIR, "mejoradas")
    segmentadas_dir = os.path.join(BASE_DIR, "segmentadas")
    anotadas_dir = os.path.join(BASE_DIR, "anotadas")

    os.makedirs(mejoradas_dir, exist_ok=True)
    os.makedirs(segmentadas_dir, exist_ok=True)
    os.makedirs(anotadas_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PROCESANDO {len(imagenes)} IMÁGENES")
    print(f"{'='*60}\n")

    imagenes_con_problemas = []

    for idx, img_path in enumerate(tqdm(imagenes, desc="Procesando"), 1):
        try:
            img = cv2.imread(img_path)
            if img is None:
                imagenes_con_problemas.append((img_path, "No se pudo leer"))
                continue

            nombre_archivo = os.path.basename(img_path)
            nombre_base = os.path.splitext(nombre_archivo)[0]

            # Mejorar calidad
            img_mejorada = mejorar_calidad(img)
            
            # Segmentar imagen
            img_segmentada, contornos = segmentar_imagen(img_mejorada)

            # Guardar imagen mejorada
            cv2.imwrite(
                os.path.join(mejoradas_dir, f"{nombre_base}_enhanced.jpg"), 
                img_mejorada
            )
            
            # Guardar imagen segmentada
            cv2.imwrite(
                os.path.join(segmentadas_dir, f"{nombre_base}_segmented.jpg"), 
                img_segmentada
            )

            # Crear y guardar imagen anotada
            img_anotada = img_mejorada.copy()
            cv2.drawContours(img_anotada, contornos, -1, (0, 255, 0), 2)
            cv2.imwrite(
                os.path.join(anotadas_dir, f"{nombre_base}_annotated.jpg"), 
                img_anotada
            )

        except Exception as e:
            imagenes_con_problemas.append((img_path, str(e)))
            print(f"\n❌ Error en {os.path.basename(img_path)}: {str(e)}")

    # Reportar problemas si existen
    if imagenes_con_problemas:
        print(f"\n⚠ ADVERTENCIAS ({len(imagenes_con_problemas)} imágenes):")
        for img, problema in imagenes_con_problemas[:5]:
            print(f"  - {os.path.basename(img)}: {problema}")
        if len(imagenes_con_problemas) > 5:
            print(f"  ... y {len(imagenes_con_problemas) - 5} más")

    print(f"\n{'='*60}")
    print("PROCESO COMPLETADO ✓")
    print(f"{'='*60}")
    print(f"\nResultados guardados en:")
    print(f"  📁 {mejoradas_dir}")
    print(f"  📁 {segmentadas_dir}")
    print(f"  📁 {anotadas_dir}")

if __name__ == "__main__":
    procesar_lote_imagenes()