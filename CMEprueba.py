import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ============================================================================
# PARÁMETROS FÍSICOS Y CONSTANTES
# ============================================================================

tr1 = 0.5  # Tiempo característico de crecimiento
td1 = 0.6  # Tiempo característico de decaimiento  
ar1 = 1    # Amplitud de crecimiento
ad1 = 3    # Amplitud de decaimiento
v01 = 0.2  # Velocidad inicial
x01 = 0.1  # Posición inicial

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
    Calcula el desplazamiento integrando la velocidad
    x(t) = x0 + ∫₀ᵗ v(s) ds
    """
    # Integramos directamente la función velocidad
    # Usamos quad con la función velocidad definida arriba
    return x01 + quad(velocidad, 0, t)[0]

def aceleracion(t):
    """
    Aceleración instantánea (es simplemente f(t))
    """
    return f(t)

# ============================================================================
# FUNCIONES DE VELOCIDAD PARA EL CAMPO VECTORIAL
# ============================================================================

def velocidad_x(theta, r, t):
    """
    Componente x de la velocidad para el campo vectorial
    Usa la velocidad del centro más componentes adicionales
    """
    # Velocidad base del centro
    v_centro = velocidad(t)
    # Componente adicional dependiente de posición
    v_extra = 0.1 * r * np.cos(theta) * t
    return v_centro + v_extra

def velocidad_radial(theta, r, t):
    """
    Componente radial de la velocidad que evoluciona con el tiempo
    """
    vy = r * t * np.cos(theta)
    vxx = velocidad_x(theta, r, t)
    return np.sqrt(vxx**2 + vy**2)

def velocidad_angular(theta, r, t):
    """
    Componente angular de la velocidad que evoluciona con el tiempo
    """
    fase = np.pi * t / 16
    return np.sin(2 * theta + fase) * (2 - r**2) * (1 + 0.2 * np.cos(fase))

def densidad(theta, r, t, densidad_fondo=0.3):
    """
    Densidad modificada para ser mayor en el borde derecho del cardioide
    """
    fase = np.pi * t / 10
    
    # COMPONENTE PRINCIPAL: MÁXIMA DENSIDAD EN EL BORDE DERECHO (theta ≈ 0)
    componente_derecha = (1 + np.cos(theta))
    
    # REFUERZO EN EL BORDE
    posicion_cardioide = desplazamiento_centro(t)
    factor_borde = np.exp(-(r - posicion_cardioide)**2)
    
    # ONDA PROPAGANTE que enfatiza el lado derecho
    onda = np.sin(3 * r - fase) * 0.5 * componente_derecha
    
    # COMPONENTE ANGULAR que refuerza el lado derecho
    angular = 0.8 * componente_derecha * np.sin(fase)
    
    # BASE con distribución radial
    base = np.exp(-r * 0.5) * (1 + 2 * componente_derecha * factor_borde)
    
    # DENSIDAD TOTAL
    densidad_total = base * (1 + onda + angular) + densidad_fondo
    
    return densidad_total

def cardioide_desplazado(theta, t, r0=0.1, theta0=0):
    """
    Genera un cardioide que se propaga siguiendo la cinemática correcta
    x(t) = x0 + ∫₀ᵗ v(s) ds, donde v(t) = v0 + ∫₀ᵗ a(s) ds
    """
    desplazamiento_cinematica = desplazamiento_centro(t) - x01  # Solo el desplazamiento desde inicial
    r_base = 0.1 + np.cos(theta)
    r = r_base + desplazamiento_cinematica
    return r

# ============================================================================
# GRÁFICAS CINEMÁTICAS CORRECTAS (POSICIÓN, VELOCIDAD, ACELERACIÓN)
# ============================================================================

# Crear array de tiempos
tiempos = np.linspace(0, 9, 100)

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
ax1.set_xlabel('Tiempo (t)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Posición (x)', fontsize=12, fontweight='bold')
ax1.set_title('POSICIÓN: x(t) = x₀ + ∫₀ᵗ v(s) ds', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# GRÁFICA 2: VELOCIDAD vs TIEMPO
ax2.plot(tiempos, velocidades, 'r-', linewidth=2.5, label='Velocidad v(t)')
ax2.set_xlabel('Tiempo (t)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Velocidad (v)', fontsize=12, fontweight='bold')
ax2.set_title('VELOCIDAD: v(t) = v₀ + ∫₀ᵗ a(s) ds', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# GRÁFICA 3: ACELERACIÓN vs TIEMPO
ax3.plot(tiempos, aceleraciones, 'g-', linewidth=2.5, label='Aceleración a(t)')
ax3.set_xlabel('Tiempo (t)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Aceleración (a)', fontsize=12, fontweight='bold')
ax3.set_title('ACELERACIÓN: a(t) = f(t)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

plt.tight_layout(pad=3.0)
plt.savefig("cinematica_corregida_cme.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# VISUALIZACIÓN DE LA CME CON CINEMÁTICA CORREGIDA
# ============================================================================

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Propagación de CME - Cinemática Corregida (f(s) = Aceleración)', 
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
    theta_dense = np.linspace(0, 2 * np.pi, 200)
    r_dense = np.linspace(0, 3.5, 120)
    THETA_dense, R_dense = np.meshgrid(theta_dense, r_dense)
    
    R_limite_dense = cardioide_desplazado(THETA_dense, t)
    mascara_dense = R_dense <= R_limite_dense
    
    # CALCULAR DENSIDAD
    dens = densidad(THETA_dense, R_dense, t, densidad_fondo=0.3)
    dens_enmascarada = np.where(mascara_dense, dens, np.nan)
    
    # Graficar mapa de densidad
    contour = ax.contourf(THETA_dense, R_dense, dens_enmascarada, 
                          levels=20, cmap='hot', alpha=0.9)
    
    # ========================================================================
    # DENSIDAD DE FONDO
    # ========================================================================
    mascara_fondo = R_dense > R_limite_dense
    dens_fondo = np.full_like(dens, 0.3)
    dens_fondo_enmascarada = np.where(mascara_fondo, dens_fondo, np.nan)
    
    ax.contourf(THETA_dense, R_dense, dens_fondo_enmascarada, 
                levels=20, cmap='Blues', alpha=0.2)
    
    # ========================================================================
    # CAMPO VECTORIAL
    # ========================================================================
    theta_vec = np.linspace(0, 2 * np.pi, 30)
    r_vec = np.linspace(0.2, 3.0, 10)
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
    ax.set_ylim([0, 3.5])
    
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