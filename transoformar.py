import os
import cv2
import numpy as np
from tkinter import Tk, filedialog
from tqdm import tqdm

# Ruta base donde guardarás las imágenes
BASE_DIR = r"C:\Users\paufu\OneDrive\Documentos\Python\TMTI\proyecto"

def seleccionar_imagenes():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilenames(
        title="Selecciona las imágenes (250 aprox)",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
    )

def mejorar_calidad(img):
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
    imagenes = seleccionar_imagenes()
    if not imagenes:
        print("No se seleccionaron imágenes.")
        return

    # Crear subcarpetas dentro del directorio base
    mejoradas_dir = os.path.join(BASE_DIR, "mejoradas")
    segmentadas_dir = os.path.join(BASE_DIR, "segmentadas")
    anotadas_dir = os.path.join(BASE_DIR, "anotadas")

    os.makedirs(mejoradas_dir, exist_ok=True)
    os.makedirs(segmentadas_dir, exist_ok=True)
    os.makedirs(anotadas_dir, exist_ok=True)

    print(f"\nProcesando {len(imagenes)} imágenes...")

    for idx, img_path in enumerate(tqdm(imagenes), 1):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_mejorada = mejorar_calidad(img)
            img_segmentada, contornos = segmentar_imagen(img_mejorada)

            nombre = os.path.splitext(os.path.basename(img_path))[0]

            cv2.imwrite(os.path.join(mejoradas_dir, f"{nombre}_enhanced.jpg"), img_mejorada)
            cv2.imwrite(os.path.join(segmentadas_dir, f"{nombre}_segmented.jpg"), img_segmentada)

            img_anotada = img_mejorada.copy()
            cv2.drawContours(img_anotada, contornos, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(anotadas_dir, f"{nombre}_annotated.jpg"), img_anotada)

        except Exception as e:
            print(f"\nError en {img_path}: {str(e)}")

    print("\nProceso completado. Resultados guardados en:")
    print(f"- {mejoradas_dir}")
    print(f"- {segmentadas_dir}")
    print(f"- {anotadas_dir}")

if __name__ == "__main__":
    procesar_lote_imagenes()
