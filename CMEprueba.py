import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARÁMETROS FÍSICOS Y CONSTANTES (ECUACIÓN ORIGINAL)
# ============================================================================

tr1 = 50      # Tiempo característico de crecimiento (segundos) - más rápido
td1 = 2000     # Tiempo característico de decaimiento (segundos)
ar1 = 0.0072    # Amplitud de crecimiento (km/s²)
ad1 = 1      # Amplitud de decaimiento (km/s²)
v01 = 40       # Velocidad inicial (km/s)
x01 = 25000    # Posición inicial (km)

# Parámetros de la CME
DENSIDAD_FONDO = 100      # Densidad del viento solar (protones/cm³)
R_CME_INICIAL = 2.0        # Radio inicial de CME (unidades arbitrarias)
FACTOR_COMPRESION = 0.3   # Factor de compresión en dirección de propagación

# ============================================================================
# FUNCIONES CINEMÁTICAS ORIGINALES (f(s) = ACELERACIÓN)
# ============================================================================

def f(s):
    """
    FUNCIÓN DE ACELERACIÓN (original)
    s: variable de integración (tiempo)
    Retorna: aceleración en el tiempo s
    """
    return (ar1 * ad1) / (ad1 * np.exp(-s / tr1) + ar1 * np.exp(s / td1))

def velocidad(t):
    """
    Calcula la velocidad integrando la aceleración f(s)
    v(t) = v0 + ∫₀ᵗ a(s) ds
    """
    if t == 0:
        return v01
    return v01 + quad(f, 0, t)[0]

def desplazamiento_centro(t):
    """
    Calcula la posición integrando la velocidad
    x(t) = x0 + ∫₀ᵗ v(s) ds
    """
    if t == 0:
        return x01
    # Integración numérica de la velocidad
    n_int = 100
    tiempos_int = np.linspace(0, t, n_int)
    velocidades_int = np.array([velocidad(ti) for ti in tiempos_int])
    integral = np.trapz(velocidades_int, tiempos_int)
    return x01 + integral

def aceleracion(t):
    """
    Aceleración instantánea
    a(t) = f(t)
    """
    return f(t)

# ============================================================================
# GRÁFICAS CINEMÁTICAS (ECUACIÓN ORIGINAL)
# ============================================================================

print("\n" + "="*80)
print("CINEMÁTICA: ECUACIÓN ORIGINAL")
print("="*80)

# Crear array de tiempos
tiempos = np.linspace(0, 36000, 500)  # en segundos (10 horas completas)

# Calcular cantidades cinemáticas
print("Calculando cinemática...")
posiciones = np.array([desplazamiento_centro(t) for t in tiempos])
velocidades = np.array([velocidad(t) for t in tiempos])
aceleraciones = np.array([aceleracion(t) for t in tiempos])

# Crear figura con 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11))

# GRÁFICA 1: POSICIÓN vs TIEMPO
ax1.plot(tiempos, posiciones, 'b-', linewidth=2.5, label='Posición x(t)')
ax1.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Posición (km)', fontsize=12, fontweight='bold')
ax1.set_title('POSICIÓN: x(t) = x₀ + ∫₀ᵗ v(s) ds', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper left')

# GRÁFICA 2: VELOCIDAD vs TIEMPO
ax2.plot(tiempos, velocidades, 'r-', linewidth=2.5, label='Velocidad v(t)')
ax2.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Velocidad (km/s)', fontsize=12, fontweight='bold')
ax2.set_title('VELOCIDAD: v(t) = v₀ + ∫₀ᵗ a(s) ds', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='upper left')
ax2.axhline(y=np.max(velocidades), color='r', linestyle='--', alpha=0.5, label=f'Máx: {np.max(velocidades):.1f} km/s')
ax2.legend(fontsize=10)

# GRÁFICA 3: ACELERACIÓN vs TIEMPO
ax3.plot(tiempos, aceleraciones, 'g-', linewidth=2.5, label='Aceleración a(t)')
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
ax3.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Aceleración (km/s²)', fontsize=12, fontweight='bold')
ax3.set_title('ACELERACIÓN: a(t) = f(t)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=11, loc='upper right')

plt.tight_layout(pad=3.0)
plt.savefig("cinematica_original.pdf", dpi=300, bbox_inches='tight')
print("✓ Gráfica cinemática guardada: cinematica_original.pdf")
plt.show()

print("\nRESUMEN CINEMÁTICO:")
print(f"  Velocidad inicial:        {v01:.2f} km/s")
print(f"  Velocidad máxima:         {np.max(velocidades):.2f} km/s")
print(f"  Velocidad final (t=6000): {velocidades[-1]:.2f} km/s")
print(f"  Aceleración inicial:      {aceleraciones[1]:.4f} km/s²")
print(f"  Aceleración máxima:       {np.max(aceleraciones):.4f} km/s²")
print(f"  Posición inicial:         {x01:.2f} km")
print(f"  Posición final (t=36000):  {posiciones[-1]:.2f} km")
print(f"  Distancia recorrida:      {posiciones[-1] - x01:.2f} km")

# ============================================================================
# VISUALIZACIÓN: CME EN COORDENADAS POLARES (SOL EN EL CENTRO)
# ============================================================================

print("\n" + "="*80)
print("VISUALIZACIÓN: PROPAGACIÓN Y EVOLUCIÓN DE CME")
print("="*80)

# Función para generar contorno de CME en coordenadas cartesianas
def contorno_cme_cartesiano(x_centro, r_base, t, n_puntos=500):
    """
    Genera puntos del contorno de la CME en coordenadas cartesianas
    
    Args:
        x_centro: posición horizontal del centro de la CME
        r_base: radio base de la CME
        t: tiempo actual (segundos)
        n_puntos: número de puntos en el contorno
    
    Returns:
        x_contorno, y_contorno: arrays con coordenadas del contorno
    """
    # Parámetro angular
    theta = np.linspace(0, 2*np.pi, n_puntos)
    
    # Radio se expande progresivamente
    r_t = r_base * (1.0 + 0.015 * t / 100.0)
    
    # Factor de deformación: se estira horizontalmente (dirección x), comprime en y
    t_norm = np.clip(t / 100.0, 0, 1)  # Normalizar tiempo entre 0 y 1
    estiramiento = 1.0 + 0.45 * t_norm      # Se estira en x
    compresion = 1.0 / np.sqrt(estiramiento)  # Se comprime en y
    
    # Generar contorno elíptico
    x_contorno = x_centro + r_t * estiramiento * np.cos(theta)
    y_contorno = r_t * compresion * np.sin(theta)
    
    return x_contorno, y_contorno

# ============================================================================
# VISUALIZACIÓN: CME EN COORDENADAS POLARES (CÍRCULO → MEDIALUNA)
# ============================================================================

print("\n" + "="*80)
print("VISUALIZACIÓN: CME EN COORDENADAS POLARES")
print("="*80)

# Parámetros de visualización
ESTADOS_TIEMPO = 10
tiempos_frames = np.linspace(0, 36000, ESTADOS_TIEMPO)

# Calcular el radio máximo esperado para ajustar límites dinámicamente
r_cme_max = R_CME_INICIAL + 0.08 * tiempos_frames[-1]
LIMITE_RADIO_MAX = r_cme_max * 1.5  # 50% de margen

# Crear figura
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Propagación de CME: Círculos Concéntricos que se Expanden (Coordenadas Polares)', 
             fontsize=18, fontweight='bold', y=0.99)

# Malla polar
theta = np.linspace(0, 2*np.pi, 360)
r = np.linspace(0, LIMITE_RADIO_MAX, 100)
THETA, R = np.meshgrid(theta, r)

for idx, t_frame in enumerate(tiempos_frames):
    print(f"  Frame {idx+1}/{ESTADOS_TIEMPO}: t = {t_frame:.0f} s ({t_frame/3600:.1f} h)", end="... ")
    
    # Radio de la CME: VINCULADO A LA VELOCIDAD REAL calculada
    # El radio se expande proporcionalmente a la velocidad e integración temporal
    v_t = velocidad(t_frame)
    r_cme = R_CME_INICIAL + (v_t / 100.0) * (t_frame / 100.0)  # Se expande con velocidad
    
    ax = plt.subplot(2, 5, idx + 1, projection='polar')
    
    # RADIO DEL CONTORNO: CÍRCULO QUE SE EXPANDE Y SE PROPAGA
    # ====================================================================
    
    # El radio es constante en todas las direcciones (círculo perfecto)
    r_contorno = r_cme * np.ones_like(THETA)
    
    # Máscara: dentro del contorno
    mascara_cme = R <= r_contorno
    
    # ====================================================================
    # DENSIDAD: MÁXIMA EN EL BORDE FRONTAL (θ≈0)
    # ====================================================================
    
    # Distribución angular: máxima en θ=0, mínima en θ=π
    dens_angular = (1.0 + np.cos(THETA))**2.5
    
    # Concentración en el borde (radial)
    r_norm = R / (r_cme + 0.1)
    dens_radial = np.exp(-4.0 * (r_norm - 0.85)**2)
    
    # Factor de dilución: la densidad disminuye conforme se expande (gradualmente)
    # Factor espacial: densidad ∝ 1/√r (por expansión)
    expansion_factor = (r_cme / R_CME_INICIAL)**0.5
    
    # Factor temporal: densidad ∝ 1/√t (disminuye con el tiempo)
    # Normalizamos el tiempo en minutos para que sea manejable
    t_norm = np.maximum(1.0, t_frame / 600.0)  # 1 es el mínimo para evitar división por cero
    time_factor = 1.0 / np.sqrt(t_norm)
    
    # Densidad combinada: decrece por expansión Y por tiempo
    densidad_diluida = 100.0 / expansion_factor * time_factor
    
    # Densidad total
    dens_cme = DENSIDAD_FONDO * densidad_diluida * dens_angular * (0.3 + dens_radial)
    
    # Aplicar máscara y limitar
    dens_plot = np.where(mascara_cme, dens_cme, DENSIDAD_FONDO)
    dens_plot = np.clip(dens_plot, DENSIDAD_FONDO, np.nanmax(dens_plot))
    
    # Aplicar escala logarítmica
    # Evitar log(0) usando máximo(valor, pequeño número)
    dens_plot_log = np.log10(np.maximum(dens_plot, DENSIDAD_FONDO/10.0))
    
    # Crear niveles válidos en escala logarítmica
    dens_min = float(np.nanmin(dens_plot_log))
    dens_max = float(np.nanmax(dens_plot_log))
    if dens_max <= dens_min:
        dens_max = dens_min + 0.1
    
    levels_dens = np.linspace(dens_min, dens_max, 100)
    levels_dens = np.sort(np.unique(levels_dens))
    if len(levels_dens) < 2:
        levels_dens = np.array([dens_min, dens_max])
    
    # Graficar densidad en escala logarítmica
    contourf = ax.contourf(THETA, R, dens_plot_log, levels=levels_dens, cmap='turbo', alpha=0.9)
    
    # ====================================================================
    # CAMPO DE VELOCIDADES
    # ====================================================================
    
    # Malla para vectores (menos densa)
    theta_vec = np.linspace(0, 2*np.pi, 22)
    r_vec = np.linspace(0.5, LIMITE_RADIO_MAX, 6)
    THETA_vec, R_vec = np.meshgrid(theta_vec, r_vec)
    
    # Máscara basada en radio isótropo
    mascara_vec = R_vec <= r_cme * 1.05
    
    # Velocidad radial: sigue la cinemática
    v_t = velocidad(t_frame)
    
    # Factor de visibilidad: máximo donde hay densidad
    factor_dens = (1.0 + np.cos(THETA_vec))**2.5
    factor_dens = np.clip(factor_dens, 0, 1)
    
    # Velocidad radial anisotrópica
    v_radial = v_t * factor_dens * (1.0 + 0.3 * np.cos(THETA_vec)**2)
    
    # Velocidad tangencial muy pequeña
    v_tangencial = 0.05 * v_t * np.sin(2*THETA_vec) * factor_dens
    
    # Normalizar
    v_mag = np.sqrt(v_radial**2 + v_tangencial**2)
    v_mag[v_mag == 0] = 1
    v_radial_norm = v_radial / v_mag
    v_tangencial_norm = v_tangencial / v_mag
    
    # Enmascarar
    v_rad_m = np.where(mascara_vec, v_radial_norm * factor_dens, np.nan)
    v_tang_m = np.where(mascara_vec, v_tangencial_norm * factor_dens, np.nan)
    
    # Convertir a coordenadas cartesianas para quiver
    U = v_rad_m * np.cos(THETA_vec) - v_tang_m * np.sin(THETA_vec)
    V = v_rad_m * np.sin(THETA_vec) + v_tang_m * np.cos(THETA_vec)
    
    # Graficar vectores
    ax.quiver(THETA_vec, R_vec, U, V, 
              scale=20, width=0.004, color='lime', alpha=0.9)
    
    # ====================================================================
    # NO MOSTRAR CONTORNO - Solo densidad rellena
    # ====================================================================
    
    # ====================================================================
    # INFORMACIÓN EN TÍTULO
    # ====================================================================
    
    x_t = desplazamiento_centro(t_frame)
    v_t = velocidad(t_frame)
    a_t = aceleracion(t_frame)
    
    titulo = f"t = {t_frame/60:.1f} min"
    
    ax.set_title(titulo, fontsize=10, fontweight='bold', pad=10)
    ax.set_ylim([0, LIMITE_RADIO_MAX])
    ax.set_rlabel_position(45)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)
    
    print("✓")

# Colorbar
plt.tight_layout(rect=[0, 0, 0.92, 0.97])
cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.75])
sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=dens_min, vmax=dens_max))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Densidad log₁₀(ρ) [log₁₀(protones/cm³)]', rotation=270, labelpad=25,
               fontsize=11, fontweight='bold')

plt.savefig("cme_evolucion_polar.pdf", dpi=300, bbox_inches='tight')
print("\n✓ Visualización guardada: cme_evolucion_polar.pdf")
plt.show()

# ============================================================================
# ANÁLISIS FINAL
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS DE CORRELACIÓN: DENSIDAD-VELOCIDAD")
print("="*80)

t_analisis = 1500
x_a = desplazamiento_centro(t_analisis)
v_a = velocidad(t_analisis)
a_a = aceleracion(t_analisis)

print(f"\nEn t = {t_analisis} s:")
print(f"  Posición:  {x_a:.0f} km")
print(f"  Velocidad: {v_a:.2f} km/s")
print(f"  Aceleración: {a_a:.4f} km/s²")
print(f"\n  La densidad máxima está en el FRENTE (θ ≈ 0)")
print(f"  donde la velocidad radial es máxima")
print(f"  Esto refleja un comportamiento físico correcto.")

print("\n" + "="*80)
print("✓ SIMULACIÓN COMPLETADA EXITOSAMENTE")
print("="*80)
