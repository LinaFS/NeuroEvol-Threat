import cv2
import os
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm

def seleccionar_opcion():
    """Pregunta al usuario si quiere procesar un video o un directorio"""
    root = Tk()
    root.withdraw()
    
    respuesta = messagebox.askyesno(
        "Modo de selección",
        "¿Deseas procesar un DIRECTORIO completo?\n\n"
        "Sí = Directorio con múltiples videos\n"
        "No = Un solo archivo de video"
    )
    return respuesta

def seleccionar_video():
    """Selecciona un solo archivo de video"""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de video",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV")]
    )
    return file_path

def seleccionar_directorio():
    """Selecciona un directorio con videos"""
    root = Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(
        title="Selecciona la carpeta con videos"
    )
    return dir_path

def obtener_videos_del_directorio(directorio):
    """Obtiene lista de archivos de video en el directorio"""
    extensiones_video = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
    videos = []
    
    for archivo in os.listdir(directorio):
        if archivo.endswith(extensiones_video):
            videos.append(os.path.join(directorio, archivo))
    
    return sorted(videos)

def extraer_frames_de_video(video_path, output_base, frame_interval=10):
    """
    Extrae frames de un video
    
    Args:
        video_path: ruta del video
        output_base: carpeta base donde guardar
        frame_interval: cada cuántos frames guardar (10 = 1 de cada 10)
    
    Returns:
        tuple: (frames_guardados, frames_totales)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base, f"{video_name}_frames")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ No se pudo abrir: {video_name}")
        return 0, 0

    # Obtener info del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_seg = total_frames / fps if fps > 0 else 0

    frame_count = 0
    saved_count = 0

    # Usar tqdm para mostrar progreso individual del video
    with tqdm(total=total_frames, desc=f"  {video_name}", leave=False, 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    
    return saved_count, frame_count

def procesar_videos(frame_interval=10):
    """
    Función principal que permite seleccionar un video o directorio
    
    Args:
        frame_interval: cada cuántos frames guardar
    """
    print(f"\n{'='*60}")
    print("EXTRACTOR DE FRAMES DE VIDEOS")
    print(f"{'='*60}\n")
    
    # Preguntar modo de operación
    modo_directorio = seleccionar_opcion()
    
    videos_a_procesar = []
    output_base = "proyecto/frames_output"
    
    if modo_directorio:
        # Modo directorio
        directorio = seleccionar_directorio()
        if not directorio:
            print("No se seleccionó ningún directorio.")
            return
        
        videos_a_procesar = obtener_videos_del_directorio(directorio)
        
        if not videos_a_procesar:
            print(f"No se encontraron videos en: {directorio}")
            return
        
        print(f"Directorio seleccionado: {directorio}")
        print(f"Videos encontrados: {len(videos_a_procesar)}\n")
        
    else:
        # Modo archivo individual
        video = seleccionar_video()
        if not video:
            print("No se seleccionó ningún video.")
            return
        
        videos_a_procesar = [video]
        print(f"Video seleccionado: {os.path.basename(video)}\n")
    
    # Crear carpeta base de salida
    os.makedirs(output_base, exist_ok=True)
    
    # Procesar videos
    print(f"{'='*60}")
    print(f"PROCESANDO {len(videos_a_procesar)} VIDEO(S)")
    print(f"{'='*60}")
    print(f"Intervalo de frames: 1 de cada {frame_interval}")
    print(f"Carpeta de salida: {output_base}\n")
    
    resultados = []
    
    for i, video_path in enumerate(videos_a_procesar, 1):
        nombre_video = os.path.basename(video_path)
        print(f"[{i}/{len(videos_a_procesar)}] Procesando: {nombre_video}")
        
        try:
            frames_guardados, frames_totales = extraer_frames_de_video(
                video_path, output_base, frame_interval
            )
            
            if frames_guardados > 0:
                print(f"  ✓ {frames_guardados} frames guardados (de {frames_totales} totales)")
                resultados.append({
                    'video': nombre_video,
                    'frames_guardados': frames_guardados,
                    'frames_totales': frames_totales,
                    'exito': True
                })
            else:
                print(f"  ⚠ No se guardaron frames")
                resultados.append({
                    'video': nombre_video,
                    'frames_guardados': 0,
                    'frames_totales': 0,
                    'exito': False
                })
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            resultados.append({
                'video': nombre_video,
                'frames_guardados': 0,
                'frames_totales': 0,
                'exito': False
            })
        
        print()  # Línea en blanco entre videos
    
    # Resumen final
    print(f"{'='*60}")
    print("RESUMEN DEL PROCESO")
    print(f"{'='*60}\n")
    
    videos_exitosos = sum(1 for r in resultados if r['exito'])
    total_frames = sum(r['frames_guardados'] for r in resultados)
    
    print(f"Videos procesados exitosamente: {videos_exitosos}/{len(videos_a_procesar)}")
    print(f"Total de frames extraídos: {total_frames}")
    print(f"\nFrames guardados en: {output_base}")
    
    # Listar carpetas creadas
    print(f"\nCarpetas generadas:")
    for r in resultados:
        if r['exito']:
            video_name = os.path.splitext(r['video'])[0]
            carpeta = os.path.join(output_base, f"{video_name}_frames")
            print(f"  📁 {carpeta} ({r['frames_guardados']} frames)")
    
    print(f"\n{'='*60}")
    print("PROCESO COMPLETADO ✓")
    print(f"{'='*60}")

if __name__ == "__main__":
    # CONFIGURACIÓN
    procesar_videos(
        frame_interval=10  # Cambiar para extraer más o menos frames
    )