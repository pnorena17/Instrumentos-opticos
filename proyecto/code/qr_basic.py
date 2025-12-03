import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from gif_to_frame import extraer_frames
from encript_image import encriptar_drpe
from encript_image import desencriptar_drpe
import generate_qr as gqr
import reconstruccion as rqr

def encriptacion_imagen_qr(frame, FILAS, COLS, radio_pupila = 0.65, graph = True):
    #Generamos la matriz con los QR de cada frame
    lista_qr = gqr.generar_lista_qrs(frame, filas=FILAS, cols=COLS, escala=2)

    imagen_desencriptada = []

    radio_pupila = min(frame.shape[0],frame.shape[1])*radio_pupila

    for qr in lista_qr:
        # Llamamos a la función de encriptar
        #radio = 400 #pixeles
        #img_cifrada, k1, k2, mascara_pupila = encriptar_drpe(frame, radio_pupila=radio)
        matriz_cifrada, k1, k2, mascara_pupila = encriptar_drpe(qr, radio_pupila)
        
        
        # Probamos desencriptar para ver si recuperamos la imagen
        #img_recuperada = desencriptar_drpe(img_cifrada, k1, k2)
        matriz_recuperada = desencriptar_drpe(matriz_cifrada, k1, k2)
        matriz_recuperada_ruidosa = np.abs(matriz_recuperada)
        
        imagen_desencriptada.append(matriz_recuperada_ruidosa)

        # Limpieza (son una prueba para lograr disminuir el ruido y por ende el radio)
        
        #matriz_norm = (matriz_recuperada_ruidosa - matriz_recuperada_ruidosa.min())/(matriz_recuperada_ruidosa.max() - matriz_recuperada_ruidosa.min())
        #matriz_limpia = np.where(matriz_norm > 0.5, 1, 0)
        
        # Verificación matemática rápida
        # Restamos la original de la recuperada.
        #error = np.mean((frame - img_recuperada)**2)
        error = np.mean((qr - matriz_recuperada_ruidosa)**2)
        #print(f"Error cuadrático medio (MSE): {error:.5f}")


    #Visualizamos  la recuperación del QR
    matriz_limpia = median_filter(matriz_recuperada, size=3) 
    imagen_final = rqr.leer_qr_individual(matriz_recuperada)
    imagen_reconstruida = rqr.reconstruir_mosaico(imagen_desencriptada)

    if graph:

        plt.figure(figsize=(12, 4))

        # QR generado
        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(lista_qr[-1]), cmap='binary', vmin=0, vmax=1)  # vmin=0, vmax=1: Asegura que el contraste sea total
        plt.title(f"Código QR ({lista_qr[-1].shape[0]}x{lista_qr[-1].shape[1]})")
        plt.axis('off') # Quitamos los números de los ejes para que se vea limpio

        # Lectura QR desencriptado
        if matriz_recuperada is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(np.abs(matriz_recuperada_ruidosa), cmap='gray')
            plt.title("QR desencriptado (ruido)")
            plt.colorbar(fraction=0.046, pad=0.04)

        # Imagen Recuperada de la lista de qr 
        if imagen_final is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(imagen_reconstruida), cmap='gray', vmin=0, vmax=1)
            plt.title(f"Lectura QR")
            plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return imagen_reconstruida