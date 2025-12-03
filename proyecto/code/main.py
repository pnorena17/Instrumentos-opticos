import numpy as np
import matplotlib.pyplot as plt
from gif_to_frame import extraer_frames
import generate_qr as gqr
import reconstruccion as rqr
import multiplexing as mux 
from play_gif import reproducir_gif
from qr_basic import encriptacion_imagen_qr

## Extraemos el video a procesar
# Ruta del archivo
#archivo_gif = r"C:\Users\user\Desktop\Universidad\Semestre 11\Instrumentos Opticos\Instrumentos-opticos\proyecto\video_test\SpAQ.gif"
archivo_gif = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\proyecto\video_test\stickman 200x200.gif"
# Creamos la lista con los frames extraidos del gif
lista_frames = extraer_frames(archivo_gif)

FILAS = 10
COLS = 10
RADIO_PUPILA = 0.65 #Proporcion

#Prueba DRPE con QR
if len(lista_frames) > 0:

    gif_recuperado = []

    for i, frame in enumerate(lista_frames):
        # Vamos a trabajar solo con el primer frame para probar
        print(f"Encriptando el frame {i+1}")
        frame_recuperado = encriptacion_imagen_qr(frame, FILAS, COLS, RADIO_PUPILA, graph=False)

        gif_recuperado.append(frame_recuperado)

    reproducir_gif(gif_recuperado)



"""# Prueba Multiplexing

# Configuración del Multiplexado
NUM_FRAMES = 2  # Cuántos frames vamos a sumar
RADIO_PRUEBA = None 
ESCALA_QR = 2 # Entre mas mejor escala para resistir el ruido de suma

test_qr = gqr.generar_lista_qrs(lista_frames[0], filas=FILAS, cols=COLS, escala=ESCALA_QR)

if len(test_qr) >= NUM_FRAMES:
    
    # Llamamos a la funcion externa para crear el paquete
    paquete, llaves = mux.crear_paquete_multiplexado(
        test_qr, NUM_FRAMES, radio_pupila=RADIO_PRUEBA
    )
    
    if paquete is not None:
        qrs_recuperados = []

        # Iteramos para recuperar cada uno
        for i, llaves_qr in enumerate(llaves):
            print(f"Recuperando QR {i+1}...")
            
            # A. Extraer la imagen binaria sucia del paquete
            matriz_sucia = mux.recuperar_qr_del_paquete(paquete, llaves_qr)
            
            # B. Intentar limpiar y leer el QR
            # IMPORTANTE: Aquí pasamos la matriz sucia directamente.
            
            # Simplemente guardamos la matriz sucia en la lista.
            # La función final intentará leerla.
            qrs_recuperados.append(matriz_sucia)
            
        
# 4. Reconstrucción Final
# Pasamos la lista de matrices recuperadas (aunque tengan ruido)
imagen_final = rqr.reconstruir_mosaico(qrs_recuperados)

if imagen_final is not None:
    plt.imshow(imagen_final, cmap='gray')
    plt.show()
else:
    print("No se pudo reconstruir la imagen.")"""
