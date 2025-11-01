import numpy as np
from PIL import Image
import math


#Definición de funciones a usar
def espejo ():
    reflex = np.array([[1,0],[0,1]])
    return reflex

def traslacion (distancia):
    trasl = np.array([[1,(distancia)],[0,1]])
    return trasl

def lente_delgada (foco): 
    lDelg = np.array([[1,0],[(-1/foco),1]])
    return lDelg

def encontrar_DA_y_PE(sistema):
    min_angulo = float('inf')
    diafragma_apertura = None
    pupila_entrada = {'pos': 0, 'radio': 0}
    
    pos_objeto = 0

    for i, elemento in enumerate(sistema):
        pos_imagen = elemento['pos']
        radio_imagen = elemento['diam'] / 2.0
        
        # Proyectar elemento hacia el espacio objeto
        for j in range(i - 1, -1, -1):
            lente_anterior = sistema[j]
            if lente_anterior['f'] > 0:
                do = pos_imagen - lente_anterior['pos']
                if abs(do) < 1e-9: continue # El objeto está en la lente
                
                # Manejar caso en el foco
                di = 1 / (1/lente_anterior['f'] - 1/do) if abs(do - lente_anterior['f']) > 1e-9 else float('inf')
                
                magnificacion = di / do if abs(do) > 1e-9 else 1
                
                pos_imagen = lente_anterior['pos'] + di
                radio_imagen *= magnificacion

        distancia_a_objeto = pos_imagen - pos_objeto
        # Evitar división por cero si la imagen está en el objeto
        if abs(distancia_a_objeto) < 1e-9: continue

        angulo = abs(radio_imagen / distancia_a_objeto)

        if angulo < min_angulo:
            min_angulo = angulo
            diafragma_apertura = elemento['nombre']
            # Guardamos la posición y radio de esta imagen, que es la candidata a Pupila de Entrada
            pupila_entrada['pos'] = pos_imagen
            pupila_entrada['radio'] = abs(radio_imagen)
            
    return diafragma_apertura, pupila_entrada

def encontrar_DC(sistema, pupila_entrada, nombre_AS):
    min_angulo_campo = float('inf')
    diafragma_campo = None
    
    # El punto de observación es el centro de la Pupila de Entrada
    pos_observador = pupila_entrada['pos']

    for i, elemento in enumerate(sistema):
        # Ignoramos el propio Diafragma de Apertura en este cálculo
        if elemento['nombre'] == nombre_AS:
             continue

        pos_imagen = elemento['pos']
        radio_imagen = elemento['diam'] / 2.0
        
        # Proyectar elemento hacia el espacio objeto (igual que antes)
        for j in range(i - 1, -1, -1):
            lente_anterior = sistema[j]
            if lente_anterior['f'] > 0:
                do = pos_imagen - lente_anterior['pos']
                if abs(do) < 1e-9: continue
                
                di = 1 / (1/lente_anterior['f'] - 1/do) if abs(do - lente_anterior['f']) > 1e-9 else float('inf')
                magnificacion = di / do if abs(do) > 1e-9 else 1
                
                pos_imagen = lente_anterior['pos'] + di
                radio_imagen *= magnificacion

        # 🎯 CAMBIO CLAVE: Calcular el ángulo desde el centro de la Pupila de Entrada
        distancia_a_pupila = pos_imagen - pos_observador
        if abs(distancia_a_pupila) < 1e-9: continue
        
        angulo_campo = abs(radio_imagen / distancia_a_pupila)

        if angulo_campo < min_angulo_campo:
            min_angulo_campo = angulo_campo
            diafragma_campo = elemento['nombre']
            
    return diafragma_campo

#Primero leemos la imagen en la ruta y la convierte en una matriz MxM
ruta=r"C:\Users\pauli\OneDrive\Documents\Universidad\Instrumentos-opticos\practica_1\images\Transm_E06.png"

img = Image.open(ruta).convert("L") #la convertimos a blanco y negro
arr = np.array(img)/255.0 #la normalizamos [0,1]
M_size = np.shape((arr))
if M_size[0] != M_size[1]:
    M = max(M_size[0],M_size[1])
    objeto = np.zeros((M,M)) 
    objeto[int((M-M_size[0])/2) : int((M-M_size[0])/2) + M_size[0], int((M-M_size[1])/2) : int((M-M_size[1])/2) + M_size[1]] = arr
else:
    M = M_size[0]
    objeto = arr


#Definción variables para las matrices
f = 0.500 #500 mm
d = 2*f   #d>f

#Transferencia S -> O 
matriz_cam_1 = traslacion(f)*lente_delgada(f)*traslacion(f)*espejo()*traslacion(f)*lente_delgada(f)*traslacion(f)

#Transferencia S -> U
matriz_cam_2 = traslacion(f)*lente_delgada(f)*traslacion(f/2)*espejo()*traslacion(d-(f/2))

#Definción de variables para la integral de difracción
l_BS = 0.050 #50 mm

#Para Cam1
D_L1 = 0.100 #100 mm
L1_M1 = 0.0104 #10.4 mm
L2_M1 = 0.0058 #5.8 mm
pix_cam_1 = 3.8e-6 #3.8 um
L1_cam_1 = 4640 * pix_cam_1
L2_cam_1 = 3506 * pix_cam_1

print(f"El tamaño de la apertura de la cámara 1 es: {L1_cam_1:.4f}x{L2_cam_1:.4f}")

#Para Cam2
D_L2 = 0.100 #100 mm
D_M2 = 0.050 #50 mm
pix_cam_2 = 5.2e-6 #5.2 um
L1_cam_2 = 1280 * pix_cam_2
L2_cam_2 = 1024 * pix_cam_2

print(f"El tamaño de la apertura de la cámara 2 es: {L1_cam_2:.4f}x{L2_cam_2:.4f}")

#Encontramos los diafragmas de apertura y de campo para limitar la imágen
L1_objeto = 1e-2 #Imágen de lado_1 = 1 cm
L2_objeto = 1.5e-2 #Imágen de lado_2 = 1.5 cm

# [nombre, posicion_z, diametro, distancia_focal (0 si no es lente)]
sistema_cam1 = [
    {'nombre': 'BS', 'pos': f/2, 'diam': l_BS, 'f': 0}, # BS, lo tratamos como apertura
    {'nombre': 'L1', 'pos': f, 'diam': D_L1, 'f': f},
    {'nombre': 'M1', 'pos': 2*f, 'diam': min(L1_M1, L2_M1), 'f': 0}, # Espejo, lo tratamos como apertura
    {'nombre': 'L1_retorno', 'pos': 3*f, 'diam': D_L1, 'f': f},
    {'nombre': 'BS_retorno', 'pos': f/2, 'diam': l_BS, 'f': 0}, # BS, lo tratamos como apertura
    {'nombre': 'Cam1', 'pos': 4*f, 'diam': min(L1_cam_1, L2_cam_2), 'f': 0}
]

sistema_cam2 = [
    {'nombre': 'BS', 'pos': f/2, 'diam': l_BS, 'f': 0}, # BS, lo tratamos como apertura
    {'nombre': 'M2', 'pos': (d-(f/2)), 'diam': D_M2, 'f': 0}, # Espejo, lo tratamos como apertura
    {'nombre': 'L2', 'pos': d, 'diam': D_L2, 'f': f},
    {'nombre': 'Cam2', 'pos': (d+f), 'diam': min(L1_cam_2, L2_cam_2), 'f': 0}
]

AS_cam1, PE_cam1 = encontrar_DA_y_PE(sistema_cam1)
print(f"\nEl Diafragma de Apertura para la Cámara 1 es: {AS_cam1}")
print(f"La Pupila de Entrada está en z={PE_cam1['pos']:.4f} m con radio de {PE_cam1['radio']*1000:.2f} mm")

DC_cam1 = encontrar_DC(sistema_cam1, PE_cam1, AS_cam1)
print(f"El Diafragma de Campo para la Cámara 1 es: {DC_cam1}")

AS_cam2, PE_cam2 = encontrar_DA_y_PE(sistema_cam2)
print(f"\nEl Diafragma de Apertura para la Cámara 2 es: {AS_cam2}")
print(f"La Pupila de Entrada está en z={PE_cam2['pos']:.4f} m con radio de {PE_cam2['radio']*1000:.2f} mm")

DC_cam2 = encontrar_DC(sistema_cam2, PE_cam2, AS_cam2)
print(f"El Diafragma de Campo para la Cámara 2 es: {DC_cam2}")

#Calculamos la transformación del campo inicial vista desde cada cámara