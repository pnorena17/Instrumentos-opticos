from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np

def reconstruir_mosaico_raw(mosaico_input):
    # Preparar imagen para pyzbar (0-255 uint8)
    if isinstance(mosaico_input, np.ndarray):
        # Si tu matriz usa 1=Negro, 0=Blanco:
        # Invertimos y escalamos: 1->0, 0->255
        img_uint8 = ((1 - mosaico_input) * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8)
    else:
        img = mosaico_input

    print("Escaneando QRs (esto puede tardar si la imagen es gigante)...")
    decoded_objects = decode(img)
    
    if not decoded_objects:
        print("Fallo de detección. Verifica: 1. Resolución, 2. Borde (Quiet Zone), 3. Contraste.")
        return None
        
    print(f"Códigos detectados: {len(decoded_objects)}")
    
    datos_bloques = []
    config_grid = None
    
    for obj in decoded_objects:
        try:
            texto = obj.data.decode('utf-8')
            # Parsear "IDX:H:W:TF:TC:DATOS"
            partes = texto.split(':')
            
            if len(partes) < 6: continue
            
            idx = int(partes[0])
            h = int(partes[1])
            w = int(partes[2])
            filas_tot = int(partes[3])
            cols_tot = int(partes[4])
            raw_str = partes[5] # String de '0's y '1's
            
            if config_grid is None:
                config_grid = (filas_tot, cols_tot)
            
            # Convertir string "0101" a array numpy
            # map(int, raw_str) convierte cada char a entero
            array_plano = np.array(list(map(int, raw_str)))
            
            # Validar longitud
            if len(array_plano) == h * w:
                bloque = array_plano.reshape((h, w))
                datos_bloques.append({'idx': idx, 'bloque': bloque, 'h': h, 'w': w})
            else:
                print(f"Error integridad bloque {idx}: esperados {h*w}, recibidos {len(array_plano)}")
                
        except Exception as e:
            print(f"Error procesando un QR: {e}")

    if not datos_bloques:
        return None

    # Ordenar
    datos_bloques.sort(key=lambda x: x['idx'])
    
    # Verificar si faltan bloques
    indices = [b['idx'] for b in datos_bloques]
    max_idx = max(indices)
    if len(indices) < (max_idx + 1):
        print(f"ADVERTENCIA: Faltan bloques. Detectados {len(indices)} de {max_idx+1}")

    # STITCHING (Pegado)
    tf, tc = config_grid
    
    # Crear estructura de filas
    grid_reconstruido = []
    ptr = 0
    for r in range(tf):
        fila = []
        for c in range(tc):
            # Buscar bloque con idx correcto (asumiendo orden, o búsqueda segura)
            # Dado que ordenamos la lista, intentamos sacar en orden
            if ptr < len(datos_bloques) and datos_bloques[ptr]['idx'] == (r * tc + c):
                fila.append(datos_bloques[ptr])
                ptr += 1
            else:
                # Bloque perdido: Rellenar con negro o saltar
                # Para robustez, creamos un bloque negro del tamaño esperado (estimado)
                # Esto es complejo sin saber el tamaño exacto del perdido, 
                # así que simplemente no agregamos nada y quedará un hueco en el array final.
                pass
        grid_reconstruido.append(fila)

    # Calcular tamaño final
    alto_fin = sum(max((b['h'] for b in f), default=0) for f in grid_reconstruido)
    ancho_fin = sum(b['w'] for b in grid_reconstruido[0]) if grid_reconstruido[0] else 0
    
    imagen_final = np.zeros((alto_fin, ancho_fin), dtype=int)
    
    y = 0
    for fila in grid_reconstruido:
        x = 0
        max_h_fila = 0
        for b in fila:
            h, w = b['h'], b['w']
            imagen_final[y:y+h, x:x+w] = b['bloque']
            x += w
            max_h_fila = max(max_h_fila, h)
        y += max_h_fila
        
    return imagen_final