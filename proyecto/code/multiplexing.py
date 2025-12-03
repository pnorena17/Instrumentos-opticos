import numpy as np
from scipy.ndimage import median_filter
import generate_qr as gqr
from encript_image import encriptar_drpe, desencriptar_drpe
from reconstruccion import reconstruir_mosaico

def crear_paquete_multiplexado(lista_frames, num_frames, filas, cols, escala, radio_pupila=None):
    if len(lista_frames) < num_frames:
        print(f"Error: Solo hay {len(lista_frames)} frames.")
        return None, None, None

    frames_usados = lista_frames[:num_frames]
    
    # Generar QR prueba para dimensiones
    test_qr = gqr.generar_mosaico_raw(frames_usados[0], filas=filas, cols=cols, escala=escala)
    alto, ancho = test_qr.shape
    
    paquete_optico = np.zeros((alto, ancho), dtype=complex)
    banco_llaves = []

    print(f"--- Multiplexando {num_frames} frames ---")

    for i, frame in enumerate(frames_usados):
        matriz_qr = gqr.generar_mosaico_raw(frame, filas=filas, cols=cols, escala=escala)
        
        # Encriptar
        campo_encriptado, k1, k2, _ = encriptar_drpe(matriz_qr, radio_pupila=radio_pupila)
        
        # Suma Coherente
        paquete_optico += campo_encriptado
        banco_llaves.append({'k1': k1, 'k2': k2})
        print(f"   > Frame {i+1} sumado.")

    return paquete_optico, banco_llaves, frames_usados

def recuperar_y_limpiar_frame(paquete_optico, llaves, filas, cols):
    """
    Recupera el frame aplicando:
    1. Desencriptación
    2. Normalización robusta
    3. FILTRO DE MEDIANA (Elimina ruido speckle)
    4. Barrido de umbrales
    """
    k1 = llaves['k1']
    k2 = llaves['k2']

    # Desencriptar
    campo_recuperado = desencriptar_drpe(paquete_optico, k1, k2)
    img_ruidosa = np.abs(campo_recuperado)

    # Normalización con recorte de picos (Clip de percentiles)
    # Esto ayuda si hay un punto brillante 'loco' que oscurece todo lo demas
    p1 = np.percentile(img_ruidosa, 1)
    p99 = np.percentile(img_ruidosa, 99)
    img_clipped = np.clip(img_ruidosa, p1, p99)
    img_norm = (img_clipped - img_clipped.min()) / (img_clipped.max() - img_clipped.min())

    # Barrido de mbrames + filtro mediana
    
    # Probamos varios niveles de oscuridad
    umbrales = np.arange(0.40, 0.65, 0.05) 
    
    for umbral in umbrales:
        # Binarizar
        matriz_binaria = np.where(img_norm > umbral, 1, 0)
        
        # Filtro mediana
        # size=3 significa que mira cuadritos de 3x3 y quita el ruido del centro
        matriz_limpia = median_filter(matriz_binaria, size=3) 
        
        # Intento Directo
        resultado = reconstruir_mosaico(matriz_limpia)
        if resultado is not None:
            return resultado, matriz_limpia, f"Recuperado (Umbral {umbral:.2f})"

        # Intento Invertido (Por si la fase queda al revés)
        matriz_inv = 1 - matriz_limpia
        resultado = reconstruir_mosaico_raw(matriz_inv)
        if resultado is not None:
            return resultado, matriz_inv, f"Recuperado (Invertido, Umbral {umbral:.2f})"
            
    # Si falla todo, devolvemos la última prueba para que veas el error
    return None, median_filter(np.where(img_norm > 0.5, 1, 0), size=3), "FALLO POR RUIDO"