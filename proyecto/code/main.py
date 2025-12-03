import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from gif_to_frame import extraer_frames
from encript_image import encriptar_drpe
from encript_image import desencriptar_drpe
import generate_qr as gqr
import reconstruccion as rqr
from qr_basic import encriptacion_imagen_qr
from play_gif import reproducir_gif

#import multiplexing as mux 

## Extraemos el video a procesar
# Ruta del archivo
#archivo_gif = r"C:\Users\user\Desktop\Universidad\Semestre 11\Instrumentos Opticos\Instrumentos-opticos\proyecto\video_test\SpAQ.gif"
archivo_gif = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\proyecto\video_test\stickman 200x200.gif"
# Creamos la lista con los frames extraidos del gif
lista_frames = extraer_frames(archivo_gif)

FILAS = 10
COLS = 10
RADIO_PUPILA = 0.65 #Proporcion

if len(lista_frames) > 0:

    gif_recuperado = []

    for i, frame in enumerate(lista_frames):
        # Vamos a trabajar solo con el primer frame para probar
        print(f"Encripatando el frame {i+1}")
        frame_recuperado = encriptacion_imagen_qr(frame, FILAS, COLS, RADIO_PUPILA, graph=False)

        gif_recuperado.append(frame_recuperado)

    reproducir_gif(gif_recuperado)


"""
# Prueba Multipelxing

# Configuración del Multiplexado
NUM_FRAMES = 2  # Cuántos frames vamos a sumar
RADIO_PRUEBA = None 
ESCALA_QR = 2 # Entre mas mejor escala para resistir el ruido de suma

if len(lista_frames) >= NUM_FRAMES:
    
    # Llamamos a la funcion externa para crear el paquete
    paquete, llaves, frames_orig = mux.crear_paquete_multiplexado(
        lista_frames, NUM_FRAMES, FILAS, COLS, ESCALA_QR, radio_pupila=RADIO_PRUEBA
    )
    
    if paquete is not None:
        
        # Preparamos gráfica grande
        plt.figure(figsize=(12, 4 * NUM_FRAMES))
        
        # Iteramos para recuperar cada uno
        for i in range(NUM_FRAMES):
            print(f"Recuperando Frame {i+1}...")
            
            # Llamamos a la funcion externa de recuperacion
            frame_final, qr_limpio, estado = mux.recuperar_y_limpiar_frame(
                paquete, llaves[i], FILAS, COLS
            )
            
            # Graficamos por fila
            
            # Col 1: Original
            plt.subplot(NUM_FRAMES, 3, (i*3) + 1)
            plt.imshow(frames_orig[i], cmap='gray')
            plt.title(f"Frame {i+1} Original")
            plt.axis('off')
            
            # Col 2: QR Recuperado (Binarizado)
            plt.subplot(NUM_FRAMES, 3, (i*3) + 2)
            plt.imshow(qr_limpio, cmap='binary')
            plt.title(f"QR {i+1} Recuperado")
            plt.axis('off')
            
            # Col 3: Resultado Final
            plt.subplot(NUM_FRAMES, 3, (i*3) + 3)
            if frame_final is not None:
                plt.imshow(frame_final, cmap='gray')
                plt.title(f"Recuperado: {estado}")
            else:
                plt.text(0.5, 0.5, "FALLO", ha='center', color='red')
                plt.title("Fallo")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
        
        # Visualizar el paquete encriptado (por probar)
        plt.figure(figsize=(5,5))
        plt.imshow(np.log(np.abs(paquete) + 1), cmap='gray')
        plt.title(f"Paquete Multiplexado ({NUM_FRAMES} frames)")
        plt.axis('off')
        plt.show()

else:
    print("No hay suficientes frames en el GIF para el multiplexado solicitado.")"""