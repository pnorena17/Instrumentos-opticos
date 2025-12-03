import numpy as np
from encript_image import encriptar_drpe, desencriptar_drpe

def crear_paquete_multiplexado(lista_qr, num_qr_a_usar, radio_pupila=None):
    """
    Toma una lista de QRs y suma los primeros 'num_qr_a_usar'.
    Retorna el campo complejo total y la lista de llaves para cada uno.
    """
    # Validación básica
    if len(lista_qr) < num_qr_a_usar:
        print(f"Advertencia: Pediste {num_qr_a_usar} pero solo hay {len(lista_qr)}. Usando todos.")
        qr_usados = lista_qr
    else:
        qr_usados = lista_qr[:num_qr_a_usar]
    
    # 1. Obtener dimensiones del primer QR para crear el lienzo vacío
    alto, ancho = qr_usados[0].shape
    
    # Lienzo acumulador complejo (inicia en 0)
    paquete_optico = np.zeros((alto, ancho), dtype=complex)
    banco_llaves = []

    print(f"--- Multiplexando {len(qr_usados)} QRs ---")

    for i, matriz_qr in enumerate(qr_usados):
        # Encriptar cada QR individualmente
        # Nota: matriz_qr entra como 0s y 1s.
        campo_encriptado, k1, k2, _ = encriptar_drpe(matriz_qr, radio_pupila=radio_pupila)
        
        # SUMA COHERENTE (Multiplexado)
        paquete_optico += campo_encriptado
        
        # Guardamos las llaves específicas de este QR
        banco_llaves.append({'idx': i, 'k1': k1, 'k2': k2})
        print(f"   > QR {i} encriptado y sumado.")

    return paquete_optico, banco_llaves

def recuperar_qr_del_paquete(paquete_optico, diccionario_llaves):
    """
    Intenta sacar un QR específico del paquete mezclado usando sus llaves.
    """
    k1 = diccionario_llaves['k1']
    k2 = diccionario_llaves['k2']

    # 1. Desencriptar
    # Al usar las llaves del QR 'X', la energía de los otros QRs se dispersa como ruido
    campo_recuperado = desencriptar_drpe(paquete_optico, k1, k2)
    
    # 2. Obtener Intensidad
    img_ruidosa = np.abs(campo_recuperado)

    # 3. Normalización Simple (0 a 1)
    # Evitamos división por cero
    val_min = img_ruidosa.min()
    val_max = img_ruidosa.max()
    
    if val_max == val_min:
        img_norm = np.zeros_like(img_ruidosa)
    else:
        img_norm = (img_ruidosa - val_min) / (val_max - val_min)

    # 4. Binarización Directa (Umbral 0.5)
    # Ya no hacemos barridos complejos. Si el lector es bueno, esto basta.
    # El ruido de fondo (crosstalk) suele ser grisáceo (< 0.5) y el QR blanco (> 0.5)
    matriz_binaria = np.where(img_norm > 0.5, 1, 0)

    # Retornamos la matriz binaria sucia.
    # La función 'leer_qr_individual' de reconstruccion.py se encargará de limpiarla más.
    return matriz_binaria