import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def propagar_difracción(campo_entrada, lado_entrada, long_onda, matriz_abcd):

    # Extraer elementos de la matriz
    A, B, C, D = matriz_abcd.ravel()
    
    # Caso Sistema de Imagen (B=0) 
    if abs(B) < 1e-9:
        print("Advertencia: B=0. Esto es un sistema de lente delgada.")
        print("La propagación se convierte en un escalado geométrico.")
        # La magnificación es el elemento A
        magnificacion = A
        L_salida = abs(magnificacion) * lado_entrada
        
        # Para simular la inversión y el escalado, se requiere interpolación, por ahora, solo evitemos errores
        print(f"Solución provisional: Magnificación: {M:.2f}x, Tamaño de salida: {L_salida*100:.2f} cm")
        # Devolvemos el campo de entrada para que el código no falle.
        
        if magnificacion < 0:
            campo_salida = np.flip(campo_entrada)
        else: 
            campo_salida = campo_entrada
        return campo_salida, L_salida

    #Propagación General (B != 0)
    magnificacion = campo_entrada.shape[0]  # Tamaño de la malla (ej: 1024)
    k = 2 * np.pi / long_onda
    
    # 1. Coordenadas en el plano de entrada
    dx_entrada = lado_entrada / magnificacion
    coords_entrada = np.linspace(-lado_entrada/2, lado_entrada/2, M)
    p, q = np.meshgrid(coords_entrada, coords_entrada)

    # 2. Multiplicar por el primer factor de fase cuadrática
    fase_cuadratica_A = np.exp(1j * k * A / (2 * B) * (p**2 + q**2))
    campo_intermedio = campo_entrada * fase_cuadratica_A

    # 3. Realizar la Transformada de Fourier 2D
    campo_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(campo_intermedio)))

    # 4. Coordenadas en el plano de salida
    # El espaciado de píxeles y el tamaño del campo cambian después de la propagación
    L_salida = long_onda * abs(B) / dx_entrada
    dx_salida = long_onda * abs(B) / lado_entrada
    coords_salida = np.linspace(-L_salida/2, L_salida/2, M)
    X, Y = np.meshgrid(coords_salida, coords_salida)

    # 5. Multiplicar por el segundo factor de fase cuadrática y el factor de escala
    fase_cuadratica_D = np.exp(1j * k * D / (2 * B) * (X**2 + Y**2))
    factor_escala = (dx_entrada**2) / (1j * long_onda * B)
    
    campo_salida = factor_escala * fase_cuadratica_D * campo_fourier
    
    return campo_salida, L_salida

#Primero leemos la imagen en la ruta y la convierte en una matriz MxM
ruta=r"C:\Users\user\Desktop\Universidad\Semestre 11\Instrumentos Opticos\Instrumentos-opticos\practica_1\images\Transm_E06.png"

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
matriz_cam_1 = traslacion(f)@lente_delgada(f)@traslacion(f)@espejo()@traslacion(f)@lente_delgada(f)@traslacion(f)

#Transferencia S -> U
matriz_cam_2 = traslacion(f)@lente_delgada(f)@traslacion(f/2)@espejo()@traslacion(d-(f/2))

#Definción de variables para la integral de difracción
l_BS = 0.050 #50 mm
long_onda = 633e-9 # 633 nm

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

#Determinamos el tamaño del objeto
L_objeto = 1e-2 #Imágen cuadrada de lado = 1 cm

#Calculamos la transformación del campo inicial vista desde cada cámara
##Cámara 1
campo_cam1, L_cam1 = propagar_difracción(objeto, L_objeto, long_onda, matriz_cam_1)

##Cámara 2
campo_cam2, L_cam2 = propagar_difracción(objeto, L_objeto, long_onda, matriz_cam_2)

#Visualización de resultados
intensidad_cam1 = np.abs(campo_cam1)**2
intensidad_cam2 = np.abs(campo_cam2)**2

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(intensidad_cam1, cmap='gray', extent=[-L_cam1/2*100, L_cam1/2*100, -L_cam1/2*100, L_cam1/2*100])
plt.title('Intensidad en Cámara 1')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')

plt.subplot(1, 2, 2)
plt.imshow(np.log(intensidad_cam2), cmap='gray', extent=[-L_cam2/2*100, L_cam2/2*100, -L_cam2/2*100, L_cam2/2*100])
plt.title('Intensidad en Cámara 2')
plt.xlabel('x (cm)')

plt.tight_layout()
plt.show()

