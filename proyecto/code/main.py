import numpy as np
import matplotlib.pyplot as plt
from gif_to_frame import extraer_frames
from encript_image import encriptar_drpe
from encript_image import desencriptar_drpe
import generate_qr as gqr

## Extraemos el video a procesar
# Ruta del archivo
archivo_gif = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\proyecto\video_test\SpAQ.gif" 
# Creamos la lista con los frames extraidos del gif
lista_frames = extraer_frames(archivo_gif)

if len(lista_frames) > 0:
    # Vamos a trabajar solo con el primer frame para probar
    frame = lista_frames[0]

    # Llamamos a la función de encriptar
    img_cifrada, k1, k2 = encriptar_drpe(frame)
    
    # Probamos desencriptar para ver si recuperamos la imagen
    img_recuperada = desencriptar_drpe(img_cifrada, k1, k2)
    
    # Verificación matemática rápida
    # Restamos la original de la recuperada.
    error = np.mean((frame - img_recuperada)**2)
    print(f"Error cuadrático medio (MSE): {error:.5f}")

    # 4. Graficar
    plt.figure(figsize=(15, 5))

    # Imagen Original
    plt.subplot(1, 3, 1)
    plt.imshow(frame, cmap='gray')
    plt.title("Original")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Imagen Encriptada 
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(img_cifrada), cmap='gray') 
    plt.title("Encriptada (Ruido)")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Imagen Recuperada
    plt.subplot(1, 3, 3)
    plt.imshow(img_recuperada, cmap='gray')
    plt.title("Recuperada")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

matriz = gqr.generar_qr_matriz(frame)

"""# 2. Lo visualizamos
plt.figure(figsize=(6, 6))
# cmap='binary': 0 lo pinta blanco, 1 lo pinta negro (Estándar QR)
# vmin=0, vmax=1: Asegura que el contraste sea total
plt.imshow(matriz, cmap='binary', vmin=0, vmax=1) 
plt.title(f"Código QR ({matriz.shape[0]}x{matriz.shape[1]})")
plt.axis('off') # Quitamos los números de los ejes para que se vea limpio
plt.show()"""