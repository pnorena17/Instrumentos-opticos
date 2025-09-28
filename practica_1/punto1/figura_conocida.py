import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0  # Importa la función de Bessel J0
from scipy.integrate import quad # Importa el integrador numérico
from scipy.stats import pearsonr

# --- PARÁMETROS FÍSICOS Y DE SIMULACIÓN (Comunes a ambos métodos) ---
R = 1e-3              # Radio de la abertura (ej: 1 mm)
z = 0.5              # Distancia de propagación (ej: 2 cm)
long_onda = 633e-9    # Longitud de onda (ej: 633 nm)
k = (2 * np.pi) / long_onda

# --- PARÁMETROS DE LA GRILLA NUMÉRICA ---
N = 1080              # Número de píxeles
dx = 3.7e-6           # Tamaño de píxel (3.7 um)
L = dx * N            # Dimensión del sensor/garganta
df = 1 / L            # Espaciado en el dominio de frecuencia

# ----------------------------------------------------------------------
# -------------------- 1. CÁLCULO DEL PERFIL ANALÍTICO ------------------
# ----------------------------------------------------------------------

print("Calculando perfil analítico (Integración numérica)...")

# Definir la pantalla de observación para el perfil analítico
# Usamos el rango del eje x de la simulación para la comparación directa
rho_sim_max = L / 2

N = 1080 # De la simulación
num_points_analitico = N # Usamos N para asegurar que la longitud sea 1080
rho_analitico = np.linspace(-rho_sim_max, rho_sim_max, num_points_analitico)


# Definir el integrando de la ecuación de Fresnel
def fresnel_integrand(r_prime, rho_point, k, z):
    """
    Integrando de la integral de difracción de Fresnel para una abertura circular.
    """
    # Exponencial compleja
    exp_term = np.exp(1j * k * r_prime**2 / (2 * z))
    
    # Función de Bessel J0
    bessel_term = j0(k * rho_point * r_prime / z)
    
    return bessel_term * exp_term * r_prime

# Array para guardar la intensidad analítica
intensity_analitica = np.zeros_like(rho_analitico, dtype=float)

# Prefactor constante fuera de la integral
prefactor = (2 * np.pi) / (1j * long_onda * z) * np.exp(1j * k * z)

for i, rho_point in enumerate(rho_analitico):
    # Término de fase que depende de rho
    rho_phase_term = np.exp(1j * k * rho_point**2 / (2 * z))
    
    # Realizar la integración numérica desde 0 hasta R
    integral_result, _ = quad(
        fresnel_integrand, 0, R, args=(rho_point, k, z), complex_func=True
    )
    
    # Calcular la amplitud compleja U(rho)
    U_rho = prefactor * rho_phase_term * integral_result
    
    # La intensidad es el módulo al cuadrado
    intensity_analitica[i] = np.abs(U_rho)**2

# ----------------------------------------------------------------------
# -------------------- 2. CÁLCULO DEL PERFIL SIMULADO -------------------
# ----------------------------------------------------------------------

print("Iniciando simulación de Espectro Angular...")

# Coordenadas Espaciales
n = np.arange(N) - N // 2
x = n * dx
X, Y = np.meshgrid(x, x) # Usamos x para X e Y para simplificar el perfil

# Coordenadas Espectrales
p = np.arange(N) - N // 2
fx = p * df
Fx, Fy = np.meshgrid(fx, fx)

# Abertura Circular
abertura = (X**2 + Y**2) <= R**2
U_0 = abertura.astype(np.complex128)

# Hallemos A_0 (Espectro Angular) y lo centramos
A_0 = np.fft.fftshift(np.fft.fft2(U_0))

# Hallemos A (Propagación del Espectro Angular)
argumento_raiz = (1. / long_onda) ** 2 - Fx ** 2 - Fy ** 2

# Verificamos que usemos las ondas propagantes
tmp = np.sqrt(np.abs(argumento_raiz))
kz = np.where(argumento_raiz >= 0, tmp * 2 * np.pi, 1j * tmp * 2 * np.pi)

H = np.exp(1j * z * kz)
A = A_0 * H
A_ishift = np.fft.ifftshift(A)

# Hallemos el campo de salida U
U_simulado = np.fft.ifft2(A_ishift)

# Hallemos la intensidad simulada
intensidad_simulada_2D = np.abs(U_simulado)**2

# Extraemos el perfil central (a lo largo del eje x, y=0)
center_index = N // 2
perfil_simulado = intensidad_simulada_2D[center_index, :]

print("Simulación completada.")

# ----------------------------------------------------------------------
# -------------------- 3. NORMALIZACIÓN Y GRÁFICO FINAL -----------------
# ----------------------------------------------------------------------

# Normalizamos ambos perfiles para que su máximo sea 1
perfil_simulado_norm = perfil_simulado / np.max(perfil_simulado)
perfil_analitico_norm = intensity_analitica / np.max(intensity_analitica)

correlation_coefficient, p_value = pearsonr(
    perfil_simulado_norm, 
    perfil_analitico_norm
)

print("\n--- Validación de Correlación ---")
print(f"Coeficiente de Correlación de Pearson (ρ): {correlation_coefficient:.5f}")

# Crear la gráfica con los dos perfiles
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Dibujamos los perfiles normalizados
# Las coordenadas del perfil simulado son 'x'
ax.plot(x * 1e3, perfil_analitico_norm, 'r-', linewidth=3, label='Perfil Analítico (Integración)')
ax.plot(x * 1e3, perfil_simulado_norm, 'b--', linewidth=2, label='Perfil Simulación (Espectro Angular)')

# Personalización del gráfico
ax.set_title(f"Comparación de Perfiles de Difracción de Fresnel", fontsize=16)
ax.set_xlabel("Posición radial x (mm)", fontsize=12)
ax.set_ylabel("Intensidad Normalizada", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=11)

# Ajustar el zoom para ver los detalles de difracción
zoom_limit = 3 * R * 1e3 
ax.set_xlim(-zoom_limit, zoom_limit)

fig.tight_layout()
plt.show()