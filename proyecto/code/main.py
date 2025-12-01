import numpy as np
import matplotlib.pyplot as plt
from gif_to_frame import extraer_frames
from encript_image import encriptar_drpe
from encript_image import desencriptar_drpe
import generate_qr as gqr
from reconstruccion import reconstruir_mosaico_raw

## Extraemos el video a procesar
# Ruta del archivo
archivo_gif = r"C:\Users\user\Desktop\Universidad\Semestre 11\Instrumentos Opticos\Instrumentos-opticos\proyecto\video_test\SpAQ.gif" 
# Creamos la lista con los frames extraidos del gif
lista_frames = extraer_frames(archivo_gif)

FILAS = 6
COLS = 8
if len(lista_frames) > 0:
    # Vamos a trabajar solo con el primer frame para probar
    frame = lista_frames[0]
    
    #Generamos la matriz con los QR de cada frame
    matriz_qr = gqr.generar_mosaico_raw(frame, filas=FILAS, cols=COLS, escala=2)

    # Llamamos a la función de encriptar
    #radio = 400 #pixeles
    #img_cifrada, k1, k2, mascara_pupila = encriptar_drpe(frame, radio_pupila=radio)
    matriz_cifrada, k1, k2, mascara_pupila = encriptar_drpe(matriz_qr, radio_pupila=1900)
    
    
    # Probamos desencriptar para ver si recuperamos la imagen
    #img_recuperada = desencriptar_drpe(img_cifrada, k1, k2)
    matriz_recuperada = desencriptar_drpe(matriz_cifrada, k1, k2)
    matriz_recuperada_ruidosa = np.abs(matriz_recuperada)
    
    # Limpieza (son una prueba para lograr disminuir el ruido y por ende el radio)
    
    #matriz_norm = (matriz_recuperada_ruidosa - matriz_recuperada_ruidosa.min())/(matriz_recuperada_ruidosa.max() - matriz_recuperada_ruidosa.min())
    #matriz_limpia = np.where(matriz_norm > 0.5, 1, 0)
    
    # Verificación matemática rápida
    # Restamos la original de la recuperada.
    #error = np.mean((frame - img_recuperada)**2)
    error = np.mean((matriz_qr - matriz_recuperada_ruidosa)**2)
    #print(f"Error cuadrático medio (MSE): {error:.5f}")
    print(f"Error cuadrático medio (MSE): {error:.5f}")

    # 4. Graficar
    plt.figure(figsize=(12, 4))

    # Imagen Original
    plt.subplot(1, 3, 1)
    plt.imshow(frame, cmap='gray')
    plt.title("Original")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Imagen Encriptada 
    plt.subplot(1, 3, 2)
    #plt.imshow(np.abs(img_cifrada), cmap='gray')
    plt.imshow(np.abs(matriz_cifrada), cmap='gray')
    plt.title("Encriptada (Ruido)")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Imagen Recuperada
    plt.subplot(1, 3, 3)
    #plt.imshow(img_recuperada, cmap='gray')
    plt.imshow(matriz_recuperada_ruidosa, cmap='gray')
    plt.title("Recuperada")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

#FILAS = 6
#COLS = 8

#matriz = gqr.generar_mosaico_raw(frame, filas=FILAS, cols=COLS, escala=2)

# Lo visualizamos
plt.figure(figsize=(6, 6))
plt.imshow(matriz_qr, cmap='binary', vmin=0, vmax=1)  # vmin=0, vmax=1: Asegura que el contraste sea total
plt.title(f"Código QR ({matriz_recuperada.shape[0]}x{matriz_recuperada.shape[1]})")
plt.axis('off') # Quitamos los números de los ejes para que se vea limpio
plt.show()

imagen_final = reconstruir_mosaico_raw(matriz_recuperada_ruidosa)

# 3. Visualizar CON PROTECCIÓN
if imagen_final is not None:
    plt.figure(figsize=(10, 10))
    # cmap='binary': 0 es blanco, 1 es negro.
    # Si tus QRs salen invertidos (negro donde debe ser blanco), cambia a cmap='gray' o invierte (1 - mosaico)
    plt.imshow(imagen_final, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Mosaico Generado ({FILAS}x{COLS})")
    plt.axis('off')
    plt.show()
else:
    print("No se pudo graficar porque falló la generación de los QRs.")