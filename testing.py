import os
import imagehash
from PIL import Image

def remove_duplicate_images(folder_path, hash_size=8, threshold=0):
    """
    Elimina imágenes duplicadas (visualmente iguales) en una carpeta usando hash perceptual.
    :param folder_path: Ruta a la carpeta con imágenes.
    :param hash_size: Tamaño del hash para pHash (por defecto 8).
    :param threshold: Diferencia máxima entre hashes para considerar duplicados (0 = iguales).
    """
    hashes = {}
    deleted = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img_hash = imagehash.phash(img, hash_size=hash_size)

            # Verifica si ese hash ya existe
            duplicate_found = False
            for existing_hash in hashes:
                if abs(img_hash - existing_hash) <= threshold:
                    duplicate_found = True
                    break

            if duplicate_found:
                os.remove(file_path)
                print(f"Eliminada imagen duplicada: {filename}")
                deleted += 1
            else:
                hashes[img_hash] = file_path

        except Exception as e:
            print(f"No se pudo procesar {filename}: {e}")

    print(f"\nTotal de imágenes eliminadas: {deleted}")

# USO
folder = r"C:\Users\paufu\OneDrive\Documentos\Python\TMTI\proyecto\andy\10_frames"  # Cambia esta ruta
remove_duplicate_images(folder)
