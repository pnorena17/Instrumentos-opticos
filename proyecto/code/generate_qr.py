import segno
import numpy as np
import math
import matplotlib.pyplot as plt
from gif_to_frame import extraer_frames

def frame_a_string(frame_binario):
    ### Función que convierte la matriz del frame (0s y 1s) a una cadena de texto para que pueda entrar en el QR.

    # Aplanamos la matriz a una sola fila
    datos_planos = frame_binario.flatten()
    
    # Convertimos a string
    cadena = "".join(str(x) for x in datos_planos)
    return cadena

def dividir_imagen_en_bloques(matriz_imagen, filas_grid=4, cols_grid=5):
    alto, ancho = matriz_imagen.shape
    
    # Calculamos el tamaño de cada bloque
    alto_bloque = math.ceil(alto / filas_grid)
    ancho_bloque = math.ceil(ancho / cols_grid)
    
    bloques = []
    
    for i in range(filas_grid):
        for j in range(cols_grid):
            # Coordenadas de corte
            y_inicio = i * alto_bloque
            y_fin = min((i + 1) * alto_bloque, alto) # 'min' evita salirnos del borde
            
            x_inicio = j * ancho_bloque
            x_fin = min((j + 1) * ancho_bloque, ancho)
            
            # Cortamos el pedazo (Slicing de Numpy)
            bloque = matriz_imagen[y_inicio:y_fin, x_inicio:x_fin]
            
            bloques.append(bloque)
            
    print(f"Imagen dividida en {len(bloques)} bloques de aprox {alto_bloque}x{ancho_bloque} px.")
    return bloques

def crear_panel_qr(lista_qrs, filas=4, cols=5, padding=0):
    if not lista_qrs:
        return np.zeros((10, 10))

    # 1. Encontrar el tamaño máximo de los QRs para asignar "celdas" iguales
    # (Como los QRs pueden variar ligeramente de tamaño según la data, usamos el más grande)
    max_h = max(qr.shape[0] for qr in lista_qrs)
    max_w = max(qr.shape[1] for qr in lista_qrs)
    
    # Tamaño de la celda (QR + padding)
    cell_h = max_h + padding
    cell_w = max_w + padding
    
    # 2. Crear el lienzo vacío (negro = 0)
    alto_total = filas * cell_h
    ancho_total = cols * cell_w
    panel = np.zeros((alto_total, ancho_total), dtype=int)
    
    # 3. Pegar cada QR en su posición
    for k, qr in enumerate(lista_qrs):
        if k >= filas * cols: break # Seguridad por si hay más QRs que celdas
        
        # Calcular fila (i) y columna (j) en la rejilla
        i = k // cols
        j = k % cols
        
        # Coordenadas pixel
        y = i * cell_h
        x = j * cell_w
        
        # Dimensiones de este QR específico
        h, w = qr.shape
        
        # Pegamos el QR en el panel
        panel[y : y + h, x : x + w] = qr
        
    return panel

def generar_qr_matriz(imagen_a_codificar, escala=1):

    # Dividimos la imágen en bloques
    lista_bloques = dividir_imagen_en_bloques(imagen_a_codificar)

    qr_generados = []

    for i, bloque in enumerate(lista_bloques):
        # Aplanamos
        datos_planos = bloque.flatten().astype(int) # Aseguramos que sean int (0 o 1)
        
        # Empaquetamos: Convierte cada 8 items (0s y 1s) en 1 Byte real
        datos_packed = np.packbits(datos_planos)
        
        # Convertimos a bytes puros para Segno
        payload_bytes = datos_packed.tobytes()
        
        # Crear el encabezado.
        # Como el payload son BYTES, el encabezado también debe ser BYTES.
        # Formato: "INDICE|ANCHO|ALTO|" para saber reconstruir ese pedazo exacto
        h, w = bloque.shape
        header_str = f"{i}|{h}|{w}|"
        header_bytes = header_str.encode('utf-8')
        
        # Unimos encabezado + datos binarios comprimidos
        contenido_completo = header_bytes + payload_bytes
        
        try:
            # Creamos el QR. Segno detecta automáticamente que son bytes.
            qr = segno.make(contenido_completo, micro=False, error='H')
            
            #Le quitamos el borde al QR
            iterador_sin_borde = qr.matrix_iter(border=0)

            # Extraer matriz
            matriz_qr = np.array([list(fila) for fila in iterador_sin_borde], dtype=int)
            
            if escala > 1:
                matriz_qr = np.repeat(np.repeat(matriz_qr, escala, axis=0), escala, axis=1)
                
            qr_generados.append(matriz_qr)
            
        except Exception as e:
            print(f"Error en el bloque {i}: {e}")
            print(f"Tamaño de datos intentado: {len(contenido_completo)} bytes")
            return [] # Paramos si falla uno
        
    mosaico_qr = crear_panel_qr(qr_generados)
        
    return mosaico_qr


archivo_gif = r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\proyecto\video_test\SpAQ.gif" 
# Creamos la lista con los frames extraidos del gif
lista_frames = extraer_frames(archivo_gif)

matriz = generar_qr_matriz(lista_frames[0])

plt.figure(figsize=(6, 6))
# cmap='binary': 0 lo pinta blanco, 1 lo pinta negro (Estándar QR)
# vmin=0, vmax=1: Asegura que el contraste sea total
plt.imshow(matriz, cmap='gray', vmin=0, vmax=1) 
plt.title(f"Código QR ({matriz.shape[0]}x{matriz.shape[1]})")
plt.axis('off') # Quitamos los números de los ejes para que se vea limpio
plt.show()