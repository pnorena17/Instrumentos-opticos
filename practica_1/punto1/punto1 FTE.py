import numpy as np
import matplotlib.pyplot as plt

#En primer lugar, definimos la abertura y longitud de onda
long_de_onda = 650*(10**(-9)) #(en metros) Usamos la longitud de onda del  rojo: 650 nm
#Por decir, usaremos de abertura un cuadrado de lado l
l = 1*(10**(-3)) #(en metros) Usamos dimesión máxima: 1 mm
z_max = (l/2)**2/(long_de_onda) #(en metros) Distancia máxima de la pantalla, para que cumpla criterio de frenel
#z = z_max - 0.050 #(En metros) La disminuimos 5 cm para evitar criticidad.
z=0.1
#PODRÍAMOS REVISAR QUE z SEA POSITIVO, POR SI ALGO

##################  MÉTODO POR FUNCION DE TRANSFERENCIA EXACTA    ######################################################33

#Creamos las variables