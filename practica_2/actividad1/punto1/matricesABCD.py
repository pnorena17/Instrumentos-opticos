import numpy as np

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

#Definción variables para las matrices
f = 0.500 #500 mm
d = 2*f   #d>f

#Transferencia S -> O 
matriz_cam_1 = traslacion(f)*lente_delgada(f)*traslacion(f)*espejo()*traslacion(f)*lente_delgada(f)*traslacion(f)

#Transferencia S -> U
matriz_cam_2 = traslacion(f)*lente_delgada(f)*traslacion(d/2)*espejo()*traslacion(d/2)

#Definción de variables para la integral de difracción
l_BS = 0.050 #50 mm

#Para Cam1
D_L1 = 0.100 #100 mm
L1_M1 = 0.0104 #10.4 mm
L2_M1 = 0.0058 #5.8 mm
pix_cam_1 = 3.8e-6 #3.8 um
L1_cam_1 = 4640 * pix_cam_1
L2_cam_1 = 3506 * pix_cam_1

#Para Cam2
D_L2 = 0.100 #100 mm
D_M2 = 0.050 #50 mm
pix_cam_2 = 5.2e-6 #5.2 um
L1_cam_2 = 1280 * pix_cam_2
L2_cam_2 = 1024 * pix_cam_2

