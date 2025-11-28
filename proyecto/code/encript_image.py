import numpy as np

def encriptar_drpe(imagen_matriz):
    # Obtenemos dimensiones
    filas, cols = imagen_matriz.shape
    
    # Crear las máscaras de fase aleatorias (Las LLAVES)
    # Valores aleatorios entre 0 y 2*pi
    fase1 = 2 * np.pi * np.random.rand(filas, cols)
    fase2 = 2 * np.pi * np.random.rand(filas, cols)
    
    # Convertimos las fases a formato complejo: e^(i * fase)
    mascara1 = np.exp(1j * fase1)
    mascara2 = np.exp(1j * fase2)
    
    ### PROCESO 4F ###
    # Plano de entrada: Imagen * Mascara 1
    plano_entrada = imagen_matriz * mascara1
    
    # Primera lente (FFT)
    plano_fourier = np.fft.fftshift(np.fft.fft2(plano_entrada))
    
    # Plano de Fourier: Multiplicar por Mascara 2 (El filtro)
    plano_filtrado = plano_fourier * mascara2
    
    # Segunda lente (IFFT) -> Imagen encriptada
    imagen_encriptada = np.fft.ifft2(np.fft.ifftshift(plano_filtrado))
    
    # Retornamos la imagen encriptada y las llaves para poder desencriptar luego
    return imagen_encriptada, mascara1, mascara2

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