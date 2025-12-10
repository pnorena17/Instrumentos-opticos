import time
t_inicio_total = time.time() #cronometro global
print("Cargando librerias...", end="")
import numpy as np
import matplotlib.pyplot as plt
from gif_to_frame import extraer_frames
import generate_qr as gqr
import reconstruccion as rqr
import multiplexing as mux 
from play_gif import reproducir_gif
from qr_basic import encriptacion_imagen_qr
from encript_image import encriptar_drpe
from encript_image import desencriptar_drpe
print(f" Listo ({time.time() - t_inicio_total:.4f} s)")


## Extraemos el video a procesar
# Ruta del archivo
archivo_gif = r"C:\Users\user\Desktop\Universidad\Semestre 11\Instrumentos Opticos\Instrumentos-opticos\proyecto\video_test\stickman 200x200.gif"
#archivo_gif = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\proyecto\video_test\stickman 200x200.gif"
print("Cargando GIF...", end="")
t_carga = time.time()
# Creamos la lista con los frames extraidos del gif
lista_frames = extraer_frames(archivo_gif)
print(f" Listo ({time.time() - t_carga:.4f} s)")

FILAS = 10
COLS = 10


# Parametros
long_onda = 633e-9 #633 nm
pixel_size_detector = 6.5e-6 # tamaño de pixel (um) Alvium 1800 U-811 with Sony IMX546
foco = 0.17 #(500 mm)
RADIO_PUPILA = 0.01
resolucion_camara = 2048

print(f"El tamaño de la pupila es {RADIO_PUPILA} m y es {(2*RADIO_PUPILA)*100/(pixel_size_detector*resolucion_camara):.2f} % de la imágen de lado {pixel_size_detector*resolucion_camara} m")

modo = 4#int(input("Ingrese el modo en que desea trabajar \n1. Una imágen normal\n2. Un frame con QR\n3. Gif con QR y DRPE\n4. Con multiplexado de un QR \n"))

t_inicio_proceso = time.time() #cronometro del proceso
if (modo == 1):
    # =======================================================
    # MODO 1: Objeto Pequeño (Stickman 200x200)
    # Objetivo: Que los ejes muestren aprox 1.3 mm de lado
    # =======================================================
    
    img_input = lista_frames[0] 
    
    # 1. Obtenemos dimensiones reales de la imagen (200 px)
    filas, cols = img_input.shape
    
    # 2. Calculamos el tamaño físico de ESTA imagen
    alto_fisico_mm = filas * pixel_size_detector * 1000  # ~1.3 mm
    ancho_fisico_mm = cols * pixel_size_detector * 1000  # ~1.3 mm
    
    # 3. Definimos la extensión física CORRECTA para los ejes
    # Centrado en 0: [-ancho/2, ancho/2, -alto/2, alto/2]
    ext_modo1 = [-ancho_fisico_mm/2, ancho_fisico_mm/2, -alto_fisico_mm/2, alto_fisico_mm/2]

    print(f"\n--- MODO 1 ---")
    print(f"Dimensiones en píxeles: {cols} x {filas}")
    print(f"Dimensiones físicas: {ancho_fisico_mm:.2f} x {alto_fisico_mm:.2f} mm")

    # Encriptación
    matriz_cifrada, k1, k2, mascara_pupila = encriptar_drpe(
                img_input, radio_pupila=RADIO_PUPILA,
                dx=pixel_size_detector, long_onda=long_onda,
                foco=foco)

    matriz_recuperada = desencriptar_drpe(matriz_cifrada, k1, k2)
    matriz_recuperada_ruidosa = np.abs(matriz_recuperada)**2

    # Graficación
    plt.figure(figsize=(12, 5))

    # Objeto Original
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(img_input), cmap='gray', extent=ext_modo1) # <--- USAMOS ext_modo1
    plt.title(f"Objeto Original\n(200px = {ancho_fisico_mm:.2f} mm)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    # Encriptado
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(matriz_cifrada), cmap='gray', extent=ext_modo1)
    plt.title("Plano de Fourier")
    plt.xlabel("x (mm)")

    # Recuperado
    plt.subplot(1, 3, 3)
    plt.imshow(matriz_recuperada_ruidosa, cmap='gray', extent=ext_modo1)
    plt.title("Imagen Recuperada")
    plt.xlabel("x (mm)")

    plt.tight_layout()
    plt.show()

elif (modo == 2):
    # =======================================================
    # MODO 2: QR Grande (Llenando el sensor 2048x2048)
    # Objetivo: Que los ejes muestren aprox 13.3 mm de lado
    # =======================================================
    
    if len(lista_frames) > 0:
        frame_prueba = lista_frames[0]
        
        # Generamos el QR forzando la resolución del sensor (2048)
        # Asegúrate de que tu función encriptacion_imagen_qr acepte 'resolucion'
        # Si no la acepta, deberás escalar la imagen manualmente antes.
        try:
            frame_recuperado, img_entrada, matriz_cifrada, matriz_recuperada = encriptacion_imagen_qr(
                frame_prueba, FILAS, COLS, 
                radio_pupila=RADIO_PUPILA,
                dx=pixel_size_detector, 
                long_onda=long_onda,
                foco=foco,
                resolucion=resolucion_camara, # <--- IMPORTANTE: 2048
                graph=False
            )
        except TypeError:
             # Fallback si tu función no ha sido actualizada
            frame_recuperado, img_entrada, matriz_cifrada, matriz_recuperada = encriptacion_imagen_qr(
                frame_prueba, FILAS, COLS, radio_pupila=RADIO_PUPILA, dx=pixel_size_detector, long_onda=long_onda, foco=foco, graph=False)

        # 1. Obtenemos dimensiones reales (deberían ser 2048 ahora)
        filas, cols = img_entrada.shape
        
        # 2. Calculamos tamaño físico
        alto_fisico_mm = filas * pixel_size_detector * 1000 # ~13.3 mm
        ancho_fisico_mm = cols * pixel_size_detector * 1000
        
        # 3. Extensión física
        ext_modo2 = [-ancho_fisico_mm/2, ancho_fisico_mm/2, -alto_fisico_mm/2, alto_fisico_mm/2]

        print(f"\n--- MODO 2 ---")
        print(f"Dimensiones en píxeles: {cols} x {filas}")
        print(f"Dimensiones físicas: {ancho_fisico_mm:.2f} x {alto_fisico_mm:.2f} mm")

            # 1. Obtenemos dimensiones reales de la imagen (200 px)
        filas, cols = lista_frames[0].shape
        
        # 2. Calculamos el tamaño físico de ESTA imagen
        alto_fisico_mm = filas * pixel_size_detector * 1000  # ~1.3 mm
        ancho_fisico_mm = cols * pixel_size_detector * 1000  # ~1.3 mm
        
        # 3. Definimos la extensión física CORRECTA para los ejes
        # Centrado en 0: [-ancho/2, ancho/2, -alto/2, alto/2]
        ext_modo1 = [-ancho_fisico_mm/2, ancho_fisico_mm/2, -alto_fisico_mm/2, alto_fisico_mm/2]

        # Graficación
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(lista_frames[0]), cmap='gray', extent=ext_modo1) # <--- USAMOS ext_modo2
        plt.title(f"Imagen original")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")

        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(img_entrada), cmap='gray', extent=ext_modo2)
        plt.title("QR generado (25/25)")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(frame_recuperado), cmap='gray', extent=ext_modo1)
        plt.title("Imagen recuperada")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")

        plt.tight_layout()
        plt.show()



elif (modo == 3):

    #Prueba DRPE con QR
    if len(lista_frames) > 0:

        frames_prueba = lista_frames[:]
        gif_recuperado = []

        for i, frame in enumerate(frames_prueba):
            # Vamos a trabajar solo con el primer frame para probar
            print(f"Encriptando el frame {i+1}")
            frame_recuperado, _, _, _ = encriptacion_imagen_qr(
                frame, FILAS, COLS, radio_pupila=RADIO_PUPILA,
                dx=pixel_size_detector, long_onda=long_onda,
                foco=foco,graph=False)

            gif_recuperado.append(frame_recuperado)

        reproducir_gif(gif_recuperado)

if (modo==4):
        
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

# Tiempo final
print(f"\nTiempo TOTAL de script: {time.time() - t_inicio_total:.4f} s")