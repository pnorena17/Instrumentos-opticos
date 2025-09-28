import numpy as np
from scipy.special import j0  # Importa la función de Bessel J0
from scipy.integrate import quad # Importa el integrador numérico
import matplotlib.pyplot as plt

# --- 1. Definir los parámetros físicos ---
R = 1e-3          # Radio de la abertura (ej: 0.5 mm)
z = 0.1             # Distancia a la pantalla (ej: 50 cm)
lambda_ = 633e-9    # Longitud de onda (ej: 633 nm, láser He-Ne)

# Constante k (número de onda)
k = 2 * np.pi / lambda_

# --- 2. Definir la pantalla de observación ---
# (Calculamos el patrón a lo largo de un radio)
rho_max = 4 * R     # Radio máximo a observar en la pantalla
num_points = 500    # Número de puntos para calcular el perfil
rho = np.linspace(-rho_max, rho_max, 2*num_points) # Array de coordenadas radiales

# --- 3. Definir el integrando de la ecuación de Fresnel ---
# Esta es la parte que va dentro de la integral
# Nota: quad integra sobre el primer argumento de la función (r_prime)
def fresnel_integrand(r_prime, rho_point, k, z):
    """
    Integrando de la integral de difracción de Fresnel para una abertura circular.
    """
    # Exponencial compleja
    exp_term = np.exp(1j * k * r_prime**2 / (2 * z))
    
    # Función de Bessel J0
    bessel_term = j0(k * rho_point * r_prime / z)
    
    return bessel_term * exp_term * r_prime

# --- 4. Calcular el patrón de difracción ---
# Array para guardar la intensidad
intensity = np.zeros_like(rho)

# Prefactor constante fuera de la integral
prefactor = (2 * np.pi) / (1j * lambda_ * z) * np.exp(1j * k * z)

for i, rho_point in enumerate(rho):
    # Término de fase que depende de rho
    rho_phase_term = np.exp(1j * k * rho_point**2 / (2 * z))
    
    # Realizar la integración numérica desde 0 hasta R
    # quad devuelve una tupla (resultado, error_estimado)
    # Como el integrando es complejo, quad devuelve (integral_real + 1j*integral_imag, ...)
    integral_result, _ = quad(
        fresnel_integrand, 0, R, args=(rho_point, k, z), complex_func=True
    )
    
    # Calcular la amplitud compleja U(rho)
    U_rho = prefactor * rho_phase_term * integral_result
    
    # La intensidad es el módulo al cuadrado de la amplitud compleja
    intensity[i] = np.abs(U_rho)**2

# --- 5. Normalizar y graficar el resultado ---
# Normalizamos la intensidad para que el máximo sea 1
intensity /= np.max(intensity)

plt.figure(figsize=(10, 6))
plt.plot(rho * 1e3, intensity, label=f'z = {z} m')
plt.title(f"Patrón de Difracción de Fresnel (Abertura Circular)")
plt.xlabel("Distancia radial en la pantalla, ρ (mm)")
plt.ylabel("Intensidad Normalizada")
plt.grid(True)
plt.legend()
plt.show()