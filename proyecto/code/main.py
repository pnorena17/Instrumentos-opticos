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
#archivo_gif = r"C:\Users\user\Desktop\Universidad\Semestre 11\Instrumentos Opticos\Instrumentos-opticos\proyecto\video_test\stickman 200x200.gif"
archivo_gif = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\proyecto\video_test\stickman 200x200.gif"
# Creamos la lista con los frames extraidos del gif
lista_frames = extraer_frames(archivo_gif)

FILAS = 15
COLS = 15


# Parametros
long_onda = 633e-9 #633 nm
pixel_size_detector = 2.74e-6 # tamaño de pixel (um) Alvium 1800 U-811 with Sony IMX546
foco = 0.17 #(500 mm)
RADIO_PUPILA = 0.03
resolucion_camara = 2048

print(f"El tamaño de la pupila es {RADIO_PUPILA} m y es {(2*RADIO_PUPILA)*100/(pixel_size_detector*resolucion_camara):.2f} % de la imágen de lado {pixel_size_detector*resolucion_camara} m")
"""
#Prueba DRPE con QR
if len(lista_frames) > 0:

    frames_prueba = lista_frames[:]
    gif_recuperado = []

    for i, frame in enumerate(frames_prueba):
        # Vamos a trabajar solo con el primer frame para probar
        print(f"Encriptando el frame {i+1}")
        frame_recuperado = encriptacion_imagen_qr(
            frame, FILAS, COLS, radio_pupila=RADIO_PUPILA,
            dx=pixel_size_detector, long_onda=long_onda,
            foco=foco,graph=False)

        gif_recuperado.append(frame_recuperado)

    reproducir_gif(gif_recuperado)

"""
# Prueba Multiplexing
# Usaremos el primer frame de la lista para probar
frame_prueba = lista_frames[0]

# Generamos un QR grande de ese frame
# Aumentamos la escala para que resista mejor el ruido del multiplexado
lista_qrs_test, pixel_logico_qr = gqr.generar_lista_qrs(frame_prueba, filas=FILAS, cols=COLS, resolucion= resolucion_camara)

# Tomamos UNO de esos QRs para dividirlo y multiplexarlo
qr_para_multiplexar = lista_qrs_test[0] 

# Dividimos el QR en 2x2 = 4 partes
FILAS_GRID = 2
COLS_GRID = 2

# Llamamos a la funcion con PARAMETROS FISICOS
paquete_optico, banco_llaves, dims_orig = mux.multiplexar_imagen_en_partes(
    qr_para_multiplexar, 
    filas_grid=FILAS_GRID, 
    cols_grid=COLS_GRID, 
    radio_pupila=RADIO_PUPILA, # Usamos la constante definida arriba (ej. 1.5e-3)
    dx=pixel_size_detector,
    long_onda=long_onda,
    foco=foco,
    logico_qr = pixel_logico_qr
)

# Desencriptamos viendo la comparativa visual
# Ponemos ver_paso_a_paso=True para ver las gráficas que pediste
imagen_rearmada = mux.desencriptar_y_reconstruir(
    paquete_optico, 
    banco_llaves, 
    dims_orig, 
    ver_paso_a_paso=True
)

plt.figure()
plt.imshow(imagen_rearmada, cmap='gray')
plt.title("QR Total Reconstruido tras Multiplexado")
plt.show()
            
datos = rqr.leer_qr_individual(imagen_rearmada, ver_debug=True)

print(datos)