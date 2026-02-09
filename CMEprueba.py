import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ============================================================================
# PARÁMETROS FÍSICOS Y CONSTANTES
# ============================================================================

tr1 = 138  # Tiempo característico de crecimiento en segundos
td1 = 1249  # Tiempo característico de decaimiento en segundos
ar1 = 0.001    # Amplitud de crecimiento en kilometros por segundo cuadrado
ad1 = 4.950    # Amplitud de decaimiento en kilometros por segundo cuadrado
v01 = 40  # Velocidad inicial en kilómetros por segundo
x01 = 25000  # Posición inicial en kilómetros

# ============================================================================
# FUNCIONES CINEMÁTICAS CORREGIDAS (f(s) = ACELERACIÓN)
# ============================================================================

def f(s):
    """
    FUNCIÓN DE ACELERACIÓN (no velocidad)
    s: variable de integración (tiempo)
    Retorna: aceleración en el tiempo s
    """
    return (ar1 * ad1) / (ad1 * np.exp(-s / tr1) + ar1 * np.exp(s / td1))

def velocidad(t):
    """
    Calcula la velocidad integrando la aceleración f(s)
    v(t) = v0 + ∫₀ᵗ a(s) ds = v0 + ∫₀ᵗ f(s) ds
    """
    return v01 + quad(f, 0, t)[0]

def desplazamiento_centro(t):
    """
    x(t) = x0 + v0*t + ∬ a(s) ds du
    """
    # Alternativa: integrar la velocidad precalculada
    tiempos = np.linspace(0, t, 200)
    velocidades = [velocidad(ti) for ti in tiempos]
    # Integración trapezoidal simple
    integral = np.trapz(velocidades, tiempos)
    return x01 + integral

def aceleracion(t):
    """
    Aceleración instantánea (es simplemente f(t))
    """
    return f(t)

# ============================================================================
# FUNCIONES DE VELOCIDAD PARA EL CAMPO VECTORIAL
# ============================================================================

def velocidad_r(theta, r, t):
    """
    Componente x de la velocidad para el campo vectorial
    Usa la velocidad del centro más componentes adicionales
    """
    # Velocidad base del centro
    v_centro = velocidad(t)
    v_extra = 0.07 * np.cos(theta)  # Componente adicional dependiente de theta, puede ajustarse para generar la dependencia radial y temporal.
    return v_centro + v_extra

def velocidad_angular(theta, r, t):
    """
    Componente angular de la velocidad que evoluciona con el tiempo
    """
    return np.sin(theta/2)

def velocidad_radial(theta, r, t):
    """
    Componente radial de la velocidad que evoluciona con el tiempo
    """
    vtheta = 2 * np.cos(theta)
    vr = velocidad_r(theta, r, t)
    return np.sqrt(vr**2 + (r * vtheta)**2)


############################################################################

densidad_fondo = 1000  # Densidad del viento solar en protones por cm³

def densidad(theta, r, t, densidad_fondo):
    """
    Densidad modificada para ser mayor en el borde derecho del cardioide
    """
    fase = np.pi * r / (t + 0.1)
    
    # COMPONENTE PRINCIPAL: MÁXIMA DENSIDAD EN EL BORDE DERECHO (theta ≈ 0), en la dirección de propagación
    componente_derecha = (1 + np.cos(theta))
    
    # REFUERZO EN EL BORDE
    posicion_cardioide = desplazamiento_centro(t)
    factor_borde = 0.1 * np.exp(-(r - posicion_cardioide)**2) #crea una especie de "borde" en la densidad, usando una gaussiana centrada en la posición del cardioide.
    
    # ONDA PROPAGANTE que enfatiza el lado derecho y hacia r mayores, gracias a la fase temporal
    onda = np.sin(fase) * componente_derecha
    
    # BASE con distribución radial
    base = np.exp(-r * 0.3) * (1 + (componente_derecha * factor_borde))
    
    # DENSIDAD TOTAL
    densidad_total = (base * (1 + onda)) + densidad_fondo #es necesario verificar la forma en que se combinan las densidades.
    
    return densidad_total

def cardioide_desplazado(theta, t, a_0=2, factor_expansion=4):
    """
    Genera un cardioide que se propaga siguiendo la cinemática correcta
    x(t) = x0 + ∫₀ᵗ v(s) ds, donde v(t) = v0 + ∫₀ᵗ a(s) ds
    """
    desplazamiento_cinematica = desplazamiento_centro(t)  # Solo el desplazamiento desde inicial

    # Expansión con el tiempo
    a = a_0 * ((1 + factor_expansion) * t)
    r_base = a * (1 + np.cos(theta))
    r = r_base + desplazamiento_cinematica
    return r

# ============================================================================
# GRÁFICAS CINEMÁTICAS CORRECTAS (POSICIÓN, VELOCIDAD, ACELERACIÓN)
# ============================================================================

# Crear array de tiempos
tiempos = np.linspace(1, 6000, 200) #en segundos

# Calcular las cantidades cinemáticas
posiciones = [desplazamiento_centro(t) for t in tiempos]
velocidades = [velocidad(t) for t in tiempos]
aceleraciones = [aceleracion(t) for t in tiempos]

# Convertir a arrays
posiciones = np.array(posiciones)
velocidades = np.array(velocidades)
aceleraciones = np.array(aceleraciones)

# Crear figura con 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# GRÁFICA 1: POSICIÓN vs TIEMPO
ax1.plot(tiempos, posiciones, 'b-', linewidth=2.5, label='Posición x(t)')
ax1.set_xlabel('Tiempo (seg)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Posición (km)', fontsize=12, fontweight='bold')
ax1.set_title('POSICIÓN: x(t) = x₀ + ∫₀ᵗ v(s) ds', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# GRÁFICA 2: VELOCIDAD vs TIEMPO
ax2.plot(tiempos, velocidades, 'r-', linewidth=2.5, label='Velocidad v(t)')
ax2.set_xlabel('Tiempo (seg)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Velocidad (km/s)', fontsize=12, fontweight='bold')
ax2.set_title('VELOCIDAD: v(t) = v₀ + ∫₀ᵗ a(s) ds', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# GRÁFICA 3: ACELERACIÓN vs TIEMPO
ax3.plot(tiempos, aceleraciones, 'g-', linewidth=2.5, label='Aceleración a(t)')
ax3.set_xlabel('Tiempo (seg)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Aceleración (km/s²)', fontsize=12, fontweight='bold')
ax3.set_title('ACELERACIÓN: a(t) = f(t)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

plt.tight_layout(pad=3.0)
plt.savefig("cinematica_corregida_cme.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# VISUALIZACIÓN DE LA CME CON CINEMÁTICA CORREGIDA
# ============================================================================
limite_gráfico = 24  # Límite radial para la gráfica


fig = plt.figure(figsize=(20, 10))
fig.suptitle('Propagación de CME', 
             fontsize=16, fontweight='bold', y=0.98)

estados_tiempo = 10

for idx in range(estados_tiempo):
    t = idx
    
    ax = plt.subplot(2, 5, idx + 1, projection='polar')
    
    # ========================================================================
    # CARDIOIDE CON CINEMÁTICA CORREGIDA
    # ========================================================================
    theta_curva = np.linspace(0, 2 * np.pi, 1000)
    r_cardioide = cardioide_desplazado(theta_curva, t)
    
    # ========================================================================
    # MALLA PARA DENSIDAD
    # ========================================================================
    theta_dense = np.linspace(0, 2 * np.pi, 360)
    r_dense = np.linspace(0, limite_gráfico, 120)
    THETA_dense, R_dense = np.meshgrid(theta_dense, r_dense)
    
    R_limite_dense = cardioide_desplazado(THETA_dense, t)
    mascara_dense = R_dense <= R_limite_dense
    
    # CALCULAR DENSIDAD
    dens = densidad(THETA_dense, R_dense, t, densidad_fondo) #la densidad de fondo es 25 protones por centímetro cúbico (valor por verificar)
    dens_enmascarada = np.where(mascara_dense, dens, np.nan)
    
    # Graficar mapa de densidad
    contour = ax.contourf(THETA_dense, R_dense, dens_enmascarada, 
                          levels=20, cmap='hot', alpha=0.9)
    
    # ========================================================================
    # DENSIDAD DE FONDO
    # ========================================================================
    mascara_fondo = R_dense > R_limite_dense
    dens_fondo = np.full_like(dens, densidad_fondo)  # Densidad de fondo constante
    dens_fondo_enmascarada = np.where(mascara_fondo, dens_fondo, np.nan)
    
    ax.contourf(THETA_dense, R_dense, dens_fondo_enmascarada, 
                levels=10, cmap='Blues', alpha=0.2)
    
    # ========================================================================
    # CAMPO VECTORIAL
    # ========================================================================
    theta_vec = np.linspace(0, 2 * np.pi, 30)
    r_vec = np.linspace(0.2, limite_gráfico, 8)
    THETA_vec, R_vec = np.meshgrid(theta_vec, r_vec)
    
    R_limite_vec = cardioide_desplazado(THETA_vec, t)
    mascara_vec = R_vec <= R_limite_vec
    
    v_r = velocidad_radial(THETA_vec, R_vec, t)
    v_theta = velocidad_angular(THETA_vec, R_vec, t)
    
    magnitud = np.sqrt(v_r**2 + v_theta**2)
    magnitud_segura = np.where(magnitud == 0, 1, magnitud)
    v_r_normalized = v_r / magnitud_segura
    v_theta_normalized = v_theta / magnitud_segura
    
    v_r_masked = np.where(mascara_vec, v_r_normalized, np.nan)
    v_theta_masked = np.where(mascara_vec, v_theta_normalized, np.nan)

    U = v_r_masked * np.cos(THETA_vec) + v_theta_masked * np.sin(THETA_vec)
    V = v_r_masked * np.sin(THETA_vec) + v_theta_masked * np.cos(THETA_vec)
    
    ax.quiver(THETA_vec, R_vec, U, V, 
              scale=20, width=0.003, color='white', alpha=0.8)
    
    # ========================================================================
    # CONTORNO DEL CARDIOIDE Y MARCADORES
    # ========================================================================
    #ax.plot(theta_curva, r_cardioide, 'cyan', linewidth=2.5, alpha=0.8)
    #ax.plot([0, 0], [0, r_cardioide[0]], 'yellow', linewidth=3, alpha=0.7, linestyle='--')
    
    # ========================================================================
    # CONFIGURACIÓN DEL SUBPLOT CON INFORMACIÓN CINEMÁTICA
    # ========================================================================
    pos_actual = desplazamiento_centro(t)
    vel_actual = velocidad(t)
    acel_actual = aceleracion(t)
    
    ax.set_title(f't = {t}\n'
                 f'x = {pos_actual:.2f}\n'
                 f'v = {vel_actual:.2f}\n'
                 f'a = {acel_actual:.2f}', 
                 fontsize=9, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim([0, limite_gráfico])
    
    # Información de densidad
    if not np.all(np.isnan(dens_enmascarada)):
        max_dens = np.nanmax(dens_enmascarada)
        ax.text(0.02, 0.98, f'ρ_max = {max_dens:.3f}', 
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Barra de color
plt.tight_layout(rect=[0, 0, 0.92, 0.96])
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contour, cax=cbar_ax)
cbar.set_label('Densidad ρ(θ,r,t)', rotation=270, labelpad=25, 
               fontsize=13, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

plt.savefig("cme_cinematica_corregida.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# TABLA DE VALORES CINEMÁTICOS CORREGIDOS
# ============================================================================

print("=" * 80)
print("CINEMÁTICA CORREGIDA - f(s) = ACELERACIÓN")
print("=" * 80)
print("Relaciones:")
print("  a(t) = f(t)")
print("  v(t) = v₀ + ∫₀ᵗ a(s) ds") 
print("  x(t) = x₀ + ∫₀ᵗ v(s) ds")
print("=" * 80)
print(f"{'t':>4} {'Posición':>10} {'Velocidad':>10} {'Aceleración':>12}")
print("-" * 80)

for t in range(0, 10):
    pos = desplazamiento_centro(t)
    vel = velocidad(t)
    acel = aceleracion(t)
    print(f"{t:4d} {pos:10.4f} {vel:10.4f} {acel:12.4f}")

print("-" * 80)

# ============================================================================
# VERIFICACIÓN DE LAS RELACIONES CINEMÁTICAS
# ============================================================================

print("\nVERIFICACIÓN NUMÉRICA:")
print(f"Posición inicial x(0): {desplazamiento_centro(0):.4f} (debe ser ≈ {x01})")
print(f"Velocidad inicial v(0): {velocidad(0):.4f} (debe ser ≈ {v01})")
print(f"Aceleración inicial a(0): {aceleracion(0):.4f}")

# Verificar que la velocidad es integral de aceleración
t_test = 5
vel_integral = v01 + quad(f, 0, t_test)[0]
print(f"\nVerificación en t={t_test}:")
print(f"  v({t_test}) calculado: {velocidad(t_test):.4f}")
print(f"  v({t_test}) por integral: {vel_integral:.4f}")
print(f"  ¿Coinciden?: {np.isclose(velocidad(t_test), vel_integral, rtol=1e-3)}")

# Añade esto después de la verificación numérica
print("\n=== ANÁLISIS FÍSICO DE LA CINEMÁTICA ===")
print(f"Aceleración máxima: {np.max(aceleraciones):.4f}")
print(f"Velocidad máxima: {np.max(velocidades):.4f}")
print(f"Desplazamiento total en t=9: {posiciones[-1]:.4f}")

# ¿Qué tiempo tarda en alcanzar el 90% de velocidad máxima?
vel_max = np.max(velocidades)
indice_90 = np.where(velocidades >= 0.9*vel_max)[0][0]
t_90 = tiempos[indice_90]
print(f"Tiempo para alcanzar 90% de v_max: {t_90:.2f}")