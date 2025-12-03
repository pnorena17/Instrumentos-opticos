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

"""#Prueba DRPE con QR
if len(lista_frames) > 0:

    gif_recuperado = []

    for i, frame in enumerate(lista_frames):
        # Vamos a trabajar solo con el primer frame para probar
        print(f"Encriptando el frame {i+1}")
        frame_recuperado = encriptacion_imagen_qr(frame, FILAS, COLS, RADIO_PUPILA, graph=False)

        gif_recuperado.append(frame_recuperado)

    reproducir_gif(gif_recuperado)"""



# Prueba Multiplexing

# Configuración del Multiplexado
NUM_FRAMES = 2  # Cuántos frames vamos a sumar
RADIO_PRUEBA = None 
ESCALA_QR = 2 # Entre mas mejor escala para resistir el ruido de suma

test_qr = gqr.generar_lista_qrs(lista_frames[0], filas=FILAS, cols=COLS, escala=ESCALA_QR)

if len(test_qr) >= NUM_FRAMES:
    
    # Llamamos a la funcion externa para crear el paquete
    paquete, llaves, dims_orig = mux.multiplexar_imagen_en_partes(
        test_qr[0], 
        filas_grid=2, 
        cols_grid=2, 
        radio_pupila=50
    )
    plt.figure()
    plt.imshow(np.log1p(np.abs(paquete)), cmap='gray')
    plt.title("Paquete Óptico")
    plt.show()

    imagen_final = mux.recuperar_y_ensamblar_imagen(paquete, llaves, dims_orig)

    # 4. Ver resultado
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(test_qr[0], cmap='gray'); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(imagen_final, cmap='gray'); plt.title("Recuperada del Mosaico")
    plt.show()
            
