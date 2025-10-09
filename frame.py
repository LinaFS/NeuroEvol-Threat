import cv2
import os
from tkinter import Tk, filedialog

def select_video_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de video",
        filetypes=[("Archivos de video", "*.mp4 *.avi *.mov *.mkv")]
    )
    return file_path

def extract_frames(video_path, frame_interval=10):
    if not video_path:
        print("No se seleccionó ningún archivo.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Carpeta de destino fija (ajústala si es necesario)
    output_base = "proyecto/andy"
    output_folder = os.path.join(output_base, f"{video_name}_frames")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Se guardaron {saved_count} frames (1 de cada {frame_interval}) en '{output_folder}'.")

# Programa principal
video_file = select_video_file()
extract_frames(video_file, frame_interval=10)
