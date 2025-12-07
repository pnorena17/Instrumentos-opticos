import numpy as np

def crear_pupila(dimensiones, radio):
    filas, cols = dimensiones
    centro_y, centro_x = filas // 2, cols // 2
    
    # Crear rejilla de coordenadas
    y, x = np.ogrid[:filas, :cols]
    
    # Ecuación del círculo: (x-cx)^2 + (y-cy)^2 <= r^2
    distancia_al_centro = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)
    
    # Pupila: 1 donde está dentro del radio, 0 donde está fuera
    pupila = (distancia_al_centro <= radio).astype(float)
    
    return pupila
    

def encriptar_drpe(imagen_matriz, radio_pupila=None, dx=None, long_onda=None, foco=None, matriz_mascara = 0):
    
    #dx es el tamaño de pixel
    # Obtenemos dimensiones
    filas, cols = imagen_matriz.shape
    
    # CREAR LLAVES 
    fase1 = 2 * np.pi * np.random.rand(filas, cols)
    fase2 = 2 * np.pi * np.random.rand(filas, cols)
    
    mascara1 = np.exp(1j * fase1)
    mascara2 = np.exp(1j * fase2)
    
    # GENERAR PUPILA (Si se solicita)

    if radio_pupila is not None:
        if dx is not None and long_onda is not None and foco is not None:
            #Calculemos el tamaño del plano en el espacio de Fourier
            N = max(filas, cols) 
            dx_fourier = (long_onda * foco)/(N*dx)
            radio_pixeles = radio_pupila /dx_fourier
            pupila = crear_pupila((filas, cols), radio_pixeles)

        else:
            pupila = crear_pupila((filas, cols), radio_pupila)
    else:
        # Si no hay radio, la pupila es todo 1 (pasa todo)
        pupila = np.ones((filas, cols))
    
    # --- PROCESO 4F ---
    
    # Plano de entrada: Imagen * Mascara 1
    plano_entrada = imagen_matriz * mascara1
    
    # Primera lente (FFT) -> Plano de Fourier
    # Usamos fftshift para que el centro óptico (frecuencia 0) quede en el medio
    plano_fourier = np.fft.fftshift(np.fft.fft2(plano_entrada))
    
    # Plano de Fourier: Multiplicar por Mascara 2 Y POR LA PUPILA
    # La pupila bloquea físicamente la luz fuera del radio
    plano_filtrado = plano_fourier * mascara2 * pupila
    
    # Segunda lente (IFFT) -> Imagen encriptada
    # Deshacemos el shift antes de la inversa
    imagen_encriptada = np.fft.ifft2(np.fft.ifftshift(plano_filtrado))
    
    # Retornamos también la pupila para visualizarla si quieres
    return imagen_encriptada, mascara1, mascara2, pupila

def desencriptar_drpe(imagen_encriptada, mascara1, mascara2):
    # Lente inversa IFFT
    plano_fourier = np.fft.fftshift(np.fft.fft2(imagen_encriptada))
    
    # Quitamos la máscara 2 (multiplicando por su conjugada)
    plano_desfiltrado = plano_fourier * np.conj(mascara2)
    
    # Volvemos al dominio espacial
    plano_entrada_recuperado = np.fft.ifft2(np.fft.ifftshift(plano_desfiltrado))
    
    # Quitamos la máscara 1
    imagen_recuperada_compleja = plano_entrada_recuperado * np.conj(mascara1)
    
    # Tomamos el módulo para ver la imagen real
    return np.abs(imagen_recuperada_compleja)