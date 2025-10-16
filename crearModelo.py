import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm

# Ruta base donde están las carpetas procesadas
BASE_DIR = r"C:\Users\paufu\OneDrive\Documentos\Python\proyecto"

def seleccionar_carpetas_procesadas():
    """Selecciona las 3 carpetas de imágenes procesadas"""
    root = Tk()
    root.withdraw()
    
    carpetas = {}
    nombres = ['mejoradas', 'segmentadas', 'anotadas']
    
    print("\n" + "="*60)
    print("SELECCIÓN DE CARPETAS")
    print("="*60)
    print("Por favor, selecciona las 3 carpetas en orden:\n")
    
    for nombre in nombres:
        print(f"Selecciona la carpeta: {nombre}")
        carpeta = filedialog.askdirectory(
            title=f"Selecciona la carpeta {nombre.upper()}"
        )
        
        if not carpeta:
            messagebox.showwarning("Advertencia", f"No se seleccionó la carpeta {nombre}")
            return None
        
        carpetas[nombre] = carpeta
        print(f"  ✓ {nombre}: {carpeta}\n")
    
    return carpetas

def detectar_objetos_en_imagen(img_path, metodo='canny', min_area=500, max_area_ratio=0.9):
    """
    Detecta objetos en una imagen ya procesada y retorna bounding boxes
    
    Args:
        img_path: ruta de la imagen
        metodo: 'canny', 'adaptativo', 'combinado'
        min_area: área mínima del contorno
        max_area_ratio: ratio máximo de área
    
    Returns:
        Lista de bounding boxes [(x, y, w, h), ...]
    """
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape[:2]
    img_area = img_height * img_width
    max_area = img_area * max_area_ratio
    
    # Aplicar método de segmentación
    if metodo == 'canny':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
    elif metodo == 'adaptativo':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
    else:  # 'combinado'
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        edges = cv2.bitwise_or(canny, adaptive)
    
    # Operaciones morfológicas
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar y extraer bounding boxes
    bounding_boxes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)
        
        # Validaciones
        if area < min_area or area > max_area:
            continue
        if perimetro == 0 or len(cnt) < 5:
            continue
        
        # Validar circularidad
        circularidad = (4 * np.pi * area) / (perimetro ** 2)
        if circularidad <= 0.01:
            continue
        
        # Validar que no sea el borde
        x, y, w, h = cv2.boundingRect(cnt)
        margin = 5
        if (x <= margin or y <= margin or 
            x + w >= img_width - margin or y + h >= img_height - margin):
            continue
        
        bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes

def obtener_imagenes_de_carpeta(carpeta):
    """Obtiene lista de imágenes de una carpeta"""
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
    imagenes = [os.path.join(carpeta, f) for f in os.listdir(carpeta)
                if f.lower().endswith(extensiones)]
    return sorted(imagenes)

def generar_csv_desde_carpetas(clase_objeto='Sospechoso',
                                metodo_segmentacion='combinado',
                                min_area=500,
                                max_area_ratio=0.9):
    """
    Genera CSV de anotaciones desde las 3 carpetas de imágenes procesadas
    
    Args:
        clase_objeto: nombre de la clase (ej: 'Sospechoso')
        metodo_segmentacion: 'canny', 'adaptativo', 'combinado'
        min_area: área mínima en píxeles²
        max_area_ratio: ratio máximo de área
    """
    # Seleccionar las 3 carpetas
    carpetas = seleccionar_carpetas_procesadas()
    if not carpetas:
        print("No se completó la selección de carpetas.")
        return
    
    # Crear carpeta dataset
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Recopilar imágenes de todas las carpetas
    print(f"\n{'='*60}")
    print(f"RECOPILANDO IMÁGENES")
    print(f"{'='*60}\n")
    
    todas_imagenes = {}
    for nombre, carpeta in carpetas.items():
        imagenes = obtener_imagenes_de_carpeta(carpeta)
        todas_imagenes[nombre] = imagenes
        print(f"  {nombre}: {len(imagenes)} imágenes")
    
    # Usar la carpeta con más imágenes como referencia
    carpeta_principal = max(todas_imagenes.items(), key=lambda x: len(x[1]))
    nombre_principal = carpeta_principal[0]
    imagenes_principales = carpeta_principal[1]
    
    print(f"\n  Usando '{nombre_principal}' como carpeta principal ({len(imagenes_principales)} imágenes)")
    
    # Preparar para procesamiento
    anotaciones = []
    estadisticas = []
    imagenes_sin_detecciones = []
    
    print(f"\n{'='*60}")
    print(f"PROCESANDO IMÁGENES")
    print(f"{'='*60}")
    print(f"Clase: '{clase_objeto}'")
    print(f"Método: {metodo_segmentacion}")
    print(f"Área mínima: {min_area}px²")
    print(f"Ratio área máxima: {max_area_ratio * 100}%\n")
    
    # Procesar cada carpeta
    for nombre_carpeta, imagenes in todas_imagenes.items():
        if len(imagenes) == 0:
            print(f"⚠ Carpeta '{nombre_carpeta}' vacía, omitiendo...\n")
            continue
        
        print(f"Procesando carpeta: {nombre_carpeta}")
        
        for img_path in tqdm(imagenes, desc=f"  {nombre_carpeta}"):
            try:
                nombre_archivo = os.path.basename(img_path)
                
                # Detectar objetos y obtener bounding boxes
                bboxes = detectar_objetos_en_imagen(
                    img_path, 
                    metodo=metodo_segmentacion,
                    min_area=min_area,
                    max_area_ratio=max_area_ratio
                )
                
                if len(bboxes) == 0:
                    if nombre_carpeta == nombre_principal:
                        imagenes_sin_detecciones.append(nombre_archivo)
                
                # Agregar cada bbox como una anotación
                for x, y, w, h in bboxes:
                    anotacion = {
                        'image_filename': nombre_archivo,
                        'class': clase_objeto,
                        'x_min': x,
                        'y_min': y,
                        'x_max': x + w,
                        'y_max': y + h
                    }
                    anotaciones.append(anotacion)
                
                # Guardar estadísticas solo de la carpeta principal
                if nombre_carpeta == nombre_principal:
                    estadisticas.append({
                        'imagen': nombre_archivo,
                        'num_detecciones': len(bboxes)
                    })
                
            except Exception as e:
                print(f"\n  ❌ Error en {os.path.basename(img_path)}: {str(e)}")
        
        print()  # Línea en blanco entre carpetas
    
    # Guardar resultados
    print(f"{'='*60}")
    print("GUARDANDO RESULTADOS")
    print(f"{'='*60}\n")
    
    if anotaciones:
        # Eliminar duplicados (misma imagen puede estar en varias carpetas)
        df_anotaciones = pd.DataFrame(anotaciones)
        
        # Agrupar por imagen y conservar detecciones únicas
        print(f"  Total anotaciones antes de eliminar duplicados: {len(df_anotaciones)}")
        df_anotaciones = df_anotaciones.drop_duplicates(
            subset=['image_filename', 'x_min', 'y_min', 'x_max', 'y_max']
        )
        print(f"  Total anotaciones después de eliminar duplicados: {len(df_anotaciones)}")
        
        # Guardar CSV
        csv_path = os.path.join(dataset_dir, "anotaciones.csv")
        df_anotaciones.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ CSV guardado: {csv_path}")
        print(f"  Total de anotaciones: {len(df_anotaciones)}")
        print(f"  Total de imágenes con detecciones: {df_anotaciones['image_filename'].nunique()}")
        print(f"  Clase: '{clase_objeto}'")
        
        # Preview
        print(f"\n  Preview (primeras 5 anotaciones):")
        print(df_anotaciones.head().to_string(index=False))
        
        # Estadísticas
        if estadisticas:
            df_stats = pd.DataFrame(estadisticas)
            print(f"\n  Estadísticas:")
            print(f"    Promedio detecciones por imagen: {df_stats['num_detecciones'].mean():.2f}")
            print(f"    Máximo detecciones en una imagen: {df_stats['num_detecciones'].max()}")
            print(f"    Mínimo detecciones en una imagen: {df_stats['num_detecciones'].min()}")
            
            # Guardar estadísticas
            stats_path = os.path.join(dataset_dir, "estadisticas.csv")
            df_stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ Estadísticas guardadas: {stats_path}")
    else:
        print("⚠ No se generaron anotaciones. Verifica los parámetros de detección.")
    
    # Reportar imágenes sin detecciones
    if imagenes_sin_detecciones:
        print(f"\n⚠ Imágenes sin detecciones: {len(imagenes_sin_detecciones)}")
        reporte_path = os.path.join(dataset_dir, "imagenes_sin_detecciones.txt")
        with open(reporte_path, 'w', encoding='utf-8') as f:
            for img in imagenes_sin_detecciones:
                f.write(f"{img}\n")
        print(f"  Reporte guardado: {reporte_path}")
    
    print(f"\n{'='*60}")
    print("PROCESO COMPLETADO ✓")
    print(f"{'='*60}")
    print(f"\n💾 Archivos guardados en: {dataset_dir}")
    print(f"💡 El archivo 'anotaciones.csv' está listo para conversión a YOLO")

if __name__ == "__main__":
    # CONFIGURACIÓN
    generar_csv_desde_carpetas(
        clase_objeto='Sospechoso',        # Clase del objeto
        metodo_segmentacion='combinado',  # 'canny', 'adaptativo', 'combinado'
        min_area=500,                     # Área mínima en píxeles²
        max_area_ratio=0.9                # Máximo 90% del área de la imagen
    )