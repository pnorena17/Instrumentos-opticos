import matplotlib.pyplot as plt
import time

def reproducir_gif(lista_frames, fps=10):
    """
    Reproduce la lista de imágenes como un video en una ventana de Matplotlib.
    """
    if not lista_frames:
        print("No hay frames para reproducir.")
        return

    # 1. Crear la figura y mostrar el primer frame estático
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Guardamos el objeto 'imagen' para actualizarlo después (es más rápido que borrar y redibujar)
    # vmin=0, vmax=1 asegura que el negro sea negro y blanco sea blanco
    imagen_plot = ax.imshow(lista_frames[0], cmap='gray', vmin=0, vmax=1)
    
    ax.axis('off')
    plt.title("Iniciando Simulación...")
    
    # Pausa inicial para que te de tiempo de ver la ventana
    plt.pause(1) 
    
    # 2. Bucle de animación
    tiempo_entre_frames = 1.0 / fps
    
    for _ in range(10):
        for i, frame in enumerate(lista_frames):
            # Actualizamos los datos de la imagen (sin cerrar la ventana)
            imagen_plot.set_data(frame)
            ax.set_title(f"Simulación Óptica - Frame {i+1}/{len(lista_frames)}")
            
            # Redibujar la ventana
            plt.draw()
            
            # Pequeña pausa para que el ojo humano vea el movimiento
            plt.pause(tiempo_entre_frames)
        
    print("Simulación finalizada.")
    plt.show() # Mantiene la ventana abierta al final