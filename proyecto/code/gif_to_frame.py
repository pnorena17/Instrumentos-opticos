import os
from PIL import Image
import numpy as np

def extraer_frames(ruta_archivo):
    # Verificar que existe el archivo
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encontró el archivo en '{ruta_archivo}'")
        return []

    # Lista donde se almacenarán las matrices de cada frame
    lista_matrices = []

    # Abrimos el GIF
    with Image.open(ruta_archivo) as im:
        index = 0
        try:
            while True:
                im.seek(index)
                

                # '1' convierte a blanco y negro puro (Binario).
                frame = im.convert('1') 
                
                # Transformación a Matriz Matemática (Numpy): Convertimos a 'int' para tener ceros y unos (0, 1).
                matriz = np.array(frame, dtype=int)
                
                # Guardamos la matriz en la lista
                lista_matrices.append(matriz)
                
                index += 1
        except EOFError:
            # Se alcanzó el final del GIF
            pass

    print(f"Se cargaron {len(lista_matrices)} frames en memoria.")
    
    # Información útil para el siguiente paso
    if len(lista_matrices) > 0:
        alto, ancho = lista_matrices[0].shape
        print(f"Resolución de los frames: {alto}x{ancho} píxeles.")
    
    return lista_matrices


#archivo_gif = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\proyecto\video_test\SpAQ.gif" 
# Ejecutamos la función
#frames_binarios = extraer_frames(archivo_gif)
