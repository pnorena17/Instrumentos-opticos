import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 

#En primer lugar, definimos la abertura y longitud de onda
long_de_onda = 650*(10**(-9)) #(en metros) Usamos la longitud de onda del  rojo: 650 nm
#Por decir, usaremos de abertura un cuadrado de lado l
l = 1*(10**(-3)) #(en metros) Usamos dimesión máxima: 1 mm
z_max = (l/2)**2/(long_de_onda) #(en metros) Distancia máxima de la pantalla, para que cumpla criterio de frenel
z = z_max - 0.050 #(En metros) La disminuimos 5 cm para evitar criticidad.

#PODRÍAMOS REVISAR QUE z SEA POSITIVO, POR SI ALGO

# Ahora, crearemos las variables que necesitamos
N_f = (l/2)**2/(long_de_onda*z) #Numero de Fresnel, deber ser mayor a 1 para que cumpla la aproximación
M = 5*N_f #Criterio aliasing: M > 4N_f
Q = 130 #Debe ser mayor a q y depende de el orden de interpolacion
N = int(Q*M) #Numero de muestras totales ? (CREO QUE TIENE QUE SER UNA POTENCIA DE 2)

#Campo de entrada
dx_entrada = l/M #Dividimos la dimensión máxima de la abertura entre el numero de muestras

#Creamos la matriz MxM centrada en (M/2,M/2)
abertura = np.ones((M,M))

x = np.linspace(-M/2, M/2 - 1, M) * dx_entrada
y = np.linspace(-M/2, M/2 - 1, M) * dx_entrada
X, Y = np.meshgrid(x, y)

#Calculamos la matriz de fase cuadrática
k = 2 * np.pi / long_de_onda
fase_cuadratica = np.exp(1j * (k / (2 * z)) * (X**2 + Y**2))

#Multiplicación punto a punto
campo_en_apertura = abertura * fase_cuadratica

padded_array = np.zeros((N, N), dtype=complex)
min_index = (N-M)//2
padded_array[min_index : min_index + M, min_index : min_index + M] = campo_en_apertura
