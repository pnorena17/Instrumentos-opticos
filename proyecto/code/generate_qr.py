import segno
import numpy as np
import math

def dividir_imagen(matriz, filas, cols):
    alto, ancho = matriz.shape
    h_bloque = math.ceil(alto / filas)
    w_bloque = math.ceil(ancho / cols)
    bloques = []
    
    for i in range(filas):
        for j in range(cols):
            y1 = i * h_bloque
            y2 = min((i + 1) * h_bloque, alto)
            x1 = j * w_bloque
            x2 = min((j + 1) * w_bloque, ancho)
            x1 = int(x1) # Aseguramos enteros
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            
            bloque = matriz[y1:y2, x1:x2]
            bloques.append(bloque)
    return bloques, h_bloque, w_bloque

def escalar_qr_a_resolucion(matriz_qr, resolucion=2048):
    """
    1. Calcula cuánto debe crecer el QR para llenar la resolución.
    2. Convierte cada píxel original en un bloque grande de píxeles.
    3. Centra el resultado en la matriz final.
    """
    h_qr, w_qr = matriz_qr.shape
    
    # 1. Calcular el Factor de Escala (Zoom)
    # Cuántas veces cabe el QR en la resolución objetivo.
    # Usamos división entera (//) para que los bloques sean perfectos.
    scale = resolucion // max(h_qr, w_qr)
    
    if scale < 1:
        print("¡El QR es más grande que la cámara! No se puede escalar.")
        return matriz_qr

    # 2. "Dividir el píxel" (Escalar)
    # np.repeat repite cada fila y columna 'scale' veces.
    # Ejemplo: Si scale=10, un píxel se vuelve un cuadro de 10x10.
    qr_gigante = np.repeat(np.repeat(matriz_qr, scale, axis=0), scale, axis=1)
    
    # 3. Centrar en el sensor (Padding de lo que sobre)
    # Como usamos división entera, sobrarán unos poquitos píxeles en el borde.
    # Ejemplo: 2048 no es divisible exacto por 21. Sobrarán bordes negros.
    h_new, w_new = qr_gigante.shape
    
    pad_y = resolucion - h_new
    pad_x = resolucion - w_new
    
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    
    # Creamos la matriz final con padding negro (constant_values=0)
    # Si tu fondo es blanco, cambia constant_values a 1
    sensor = np.pad(
        qr_gigante, 
        ((top, bottom), (left, right)), 
        mode='constant', 
        constant_values=0 
    )
    
    return sensor

def generar_lista_qrs(imagen_binaria, filas=6, cols=8, resolucion = 2048):
    """
    Genera una LISTA de matrices QR individuales.
    Cada QR contiene en su header el índice 'i' para saber su coordenada.
    """
    bloques, _, _ = dividir_imagen(imagen_binaria, filas, cols)
    qr_matrices = []
    
    print(f"Generando {len(bloques)} códigos QR individuales...")

    for i, bloque in enumerate(bloques):
        # 1. Aplanar matriz
        flat = bloque.flatten().astype(int)
        
        # 2. Convertir a string "010101..."
        payload_str = "".join(map(str, flat))
        
        h, w = bloque.shape
        
        # 3. Header: "IDX:H:W:TF:TC:DATOS"
        # IDX (i) es clave: con él calculas la fila (i // cols) y columna (i % cols)
        contenido = f"{i}:{h}:{w}:{filas}:{cols}:{payload_str}"
        
        try:
            # RECOMENDACIÓN: Usar error='H' para óptica (soporta 30% daño)
            # Usar 'L' (7%) es arriesgado con el ruido speckle, pero lo dejo a tu elección.
            qr = segno.make(contenido, error='H') 
            
            # Convertir a matriz numpy con borde
            iterador = qr.matrix_iter(border=4) 
            matriz_qr = np.array([list(row) for row in iterador], dtype=int)
            
            tamano_qr_logico = matriz_qr.shape[0]

            # Configuremos la resolución
            sobremuestreo_qr = escalar_qr_a_resolucion(matriz_qr, resolucion)
            qr_matrices.append(sobremuestreo_qr)
            
        except Exception as e:
            print(f"Error en bloque {i}: {e}")
            return None

    # Retornamos la lista directa.
    return qr_matrices, tamano_qr_logico