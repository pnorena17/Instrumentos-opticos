import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importamos tus módulos
from encript_image import encriptar_drpe, desencriptar_drpe
import generate_qr as gqr
import reconstruccion as rqr

# --- CONFIGURACIÓN GLOBAL ---
RESOLUCIONES_TEST = [100, 150, 200, 250] 
RES_OPTICA_QR = 512  
RUIDO_RANGE = np.linspace(0, 0.8, 10) # Reducido a 10 puntos para que la gráfica no se sature de texto

def calcular_grid_seguro(N):
    tamano_max_bloque = 35 
    grid_necesario = math.ceil(N / tamano_max_bloque)
    return max(grid_necesario, 2)

def run_benchmark_final():
    print("--- INICIANDO BENCHMARK VISUAL (V3 - CON ETIQUETAS) ---")
    
    datos_velocidad = []
    datos_peso = []
    
    # ======================================================
    # PARTE 1: VELOCIDAD Y PESO
    # ======================================================
    print("\n[1/2] Obteniendo datos de Velocidad y Peso...")
    
    for N in RESOLUCIONES_TEST:
        print(f"   -> Procesando {N}x{N}...")
        frame = np.random.randint(0, 2, (N, N))
        
        # A. Directo
        t0 = time.time()
        u2_dir, m1, m2, _ = encriptar_drpe(frame)
        _ = desencriptar_drpe(u2_dir, m1, m2)
        fps_direct = 1.0 / (time.time() - t0)
        peso_direct_mb = u2_dir.nbytes / (1024 * 1024)
        
        # B. QR
        grid = calcular_grid_seguro(N)
        try:
            t0 = time.time()
            res_gen = gqr.generar_lista_qrs(frame, filas=grid, cols=grid, resolucion=RES_OPTICA_QR)
            if res_gen is None or res_gen[0] is None: raise ValueError("Error Gen")
            lista_qrs = res_gen[0]

            peso_acumulado_qr = 0
            qrs_recup = []
            for qr in lista_qrs:
                u2_qr, m1_q, m2_q, _ = encriptar_drpe(qr)
                rec = desencriptar_drpe(u2_qr, m1_q, m2_q)
                qrs_recup.append(rec)
                peso_acumulado_qr += u2_qr.nbytes
                
            for rec in qrs_recup: _ = rqr.leer_qr_individual(rec)
                
            fps_qr = 1.0 / (time.time() - t0)
            peso_qr_mb = peso_acumulado_qr / (1024 * 1024)
            
        except Exception:
            fps_qr = 0.01
            peso_qr_mb = 0
        
        datos_velocidad.append({"Res": N, "FPS_Dir": fps_direct, "FPS_QR": fps_qr})
        datos_peso.append({"Res": str(N), "MB_Dir": peso_direct_mb, "MB_QR": peso_qr_mb})

    # ======================================================
    # PARTE 2: ROBUSTEZ
    # ======================================================
    print("\n[2/2] Obteniendo datos de Robustez...")
    N_fix = 140
    frame_ref = np.random.randint(0, 2, (N_fix, N_fix))
    grid_fix = calcular_grid_seguro(N_fix)
    
    mse_directo_list = []
    exito_qr_list = []
    
    try:
        u2_clean_dir, m1_d, m2_d, _ = encriptar_drpe(frame_ref)
        max_amp_dir = np.max(np.abs(u2_clean_dir))
        
        res_gen = gqr.generar_lista_qrs(frame_ref, filas=grid_fix, cols=grid_fix, resolucion=RES_OPTICA_QR)
        u2_clean_qr, m1_q, m2_q, _ = encriptar_drpe(res_gen[0][0])
        max_amp_qr = np.max(np.abs(u2_clean_qr))
        
        for nivel_ruido in RUIDO_RANGE:
            # Directo
            noise_d = (np.random.randn(*u2_clean_dir.shape) + 1j * np.random.randn(*u2_clean_dir.shape))
            rec_dir = desencriptar_drpe(u2_clean_dir + (noise_d * nivel_ruido * max_amp_dir * 0.1), m1_d, m2_d)
            rec_dir_norm = np.abs(rec_dir)**2 / np.max(np.abs(rec_dir)**2)
            mse_directo_list.append(np.mean((frame_ref - rec_dir_norm)**2))
            
            # QR
            noise_q = (np.random.randn(*u2_clean_qr.shape) + 1j * np.random.randn(*u2_clean_qr.shape))
            rec_qr = desencriptar_drpe(u2_clean_qr + (noise_q * nivel_ruido * max_amp_qr * 0.1), m1_q, m2_q)
            datos = rqr.leer_qr_individual(rec_qr)
            exito_qr_list.append(100 if datos is not None else 0)
            
    except Exception as e:
        print(f"Error Robustez: {e}")
        mse_directo_list = [0]*len(RUIDO_RANGE)
        exito_qr_list = [0]*len(RUIDO_RANGE)

    # ======================================================
    # GRAFICAR CON ETIQUETAS
    # ======================================================
    print("\nGenerando gráficas detalladas...")
    df_vel = pd.DataFrame(datos_velocidad)
    df_pes = pd.DataFrame(datos_peso)
    
    plt.figure(figsize=(18, 6)) # Hacemos la figura más ancha para que quepan los números
    
    # --- GRÁFICA 1: VELOCIDAD (FPS) ---
    ax1 = plt.subplot(1, 2, 1)
    # Linea Directa
    ax1.plot(df_vel["Res"], df_vel["FPS_Dir"], 'o-', color='tab:blue', label='Directo')
    # Etiquetas Directa
    for x, y in zip(df_vel["Res"], df_vel["FPS_Dir"]):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', color='tab:blue', fontsize=9, weight='bold')
        
    # Linea QR
    ax1.plot(df_vel["Res"], df_vel["FPS_QR"], 's--', color='tab:orange', label='Sistema QR')
    # Etiquetas QR (formato científico o con más decimales porque son pequeños)
    for x, y in zip(df_vel["Res"], df_vel["FPS_QR"]):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', color='tab:orange', fontsize=9, weight='bold')

    ax1.set_yscale('log')
    ax1.set_title("Velocidad de Procesamiento (FPS)", fontsize=12)
    ax1.set_xlabel("Resolución Input")
    ax1.set_ylabel("FPS (Log)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    # --- GRÁFICA 2: PESO (BARRAS) ---
    ax2 = plt.subplot(1, 2, 2)
    x_idx = np.arange(len(df_pes))
    w = 0.35
    
    rects1 = ax2.bar(x_idx - w/2, df_pes["MB_Dir"], w, label='Directo', color='tab:blue', alpha=0.8)
    rects2 = ax2.bar(x_idx + w/2, df_pes["MB_QR"], w, label='QR', color='tab:orange', alpha=0.8)
    
    # ETIQUETAS AUTOMÁTICAS EN BARRAS
    ax2.bar_label(rects1, padding=3, fmt='%.1f MB', fontsize=8, color='tab:blue', weight='bold')
    # Para las barras naranjas, rotamos el texto si son muy altas o usamos fmt
    ax2.bar_label(rects2, padding=3, fmt='%.0f MB', fontsize=8, color='tab:orange', weight='bold')
    
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(df_pes["Res"])
    ax2.set_title("Peso de Transmisión (MB)", fontsize=12)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- GRÁFICA 3: ROBUSTEZ ---
    ax3 = plt.subplot(1, 2, 3)
    
    # Eje Izquierdo (MSE)
    line1 = ax3.plot(RUIDO_RANGE, mse_directo_list, 'o-', color='tab:blue', label='Error Directo', markersize=4)
    ax3.set_xlabel('Nivel de Ruido')
    ax3.set_ylabel('')
    # Etiquetas selectivas (cada 2 puntos para no saturar)
    for i in range(0, len(RUIDO_RANGE), 2): 
        val = mse_directo_list[i]
        ax3.annotate(f'{val:.2f}', (RUIDO_RANGE[i], val), xytext=(-10, 5), textcoords='offset points', color='tab:blue', fontsize=8)

    # Eje Derecho (Éxito)
    ax3b = ax3.twinx()
    line2 = ax3b.plot(RUIDO_RANGE, exito_qr_list, 's--', color='tab:orange', label='Éxito QR', markersize=4)
    ax3b.set_ylabel('Éxito (%)', color='tab:orange')
    ax3b.set_ylim(-10, 120)
    
    # Etiquetar el punto de caída del QR (donde deja de ser 100)
    for i, val in enumerate(exito_qr_list):
        if i > 0 and exito_qr_list[i] < 100 and exito_qr_list[i-1] == 100:
             ax3b.annotate(f'Caída\nRuido={RUIDO_RANGE[i]:.1f}', (RUIDO_RANGE[i], val), 
                           xytext=(20, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"), color='red')
             break
    
    ax3.set_title("Resistencia al Ruido", fontsize=12)
    
    # Leyenda combinada
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='center right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark_final()