import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# PARÁMETROS FÍSICOS
tr1, td1 = 138, 1249          # Tiempos característicos (s)
ar1, ad1 = 0.001, 4.950        # Amplitudes (km/s²)
v01, x01 = 40, 25000          # Velocidad (km/s) y posición inicial (km)
DENSIDAD_FONDO = 100          # Densidad base (protones/cm³)
R_CME_INICIAL = 2.0           # Radio inicial de CME

def f(s):
    """Aceleración: a(t) = (ar1·ad1) / (ad1·exp(-t/tr1) + ar1·exp(t/td1))"""
    return (ar1 * ad1) / (ad1 * np.exp(-s / tr1) + ar1 * np.exp(s / td1))

def velocidad(t):
    """v(t) = v0 + ∫₀ᵗ a(s) ds"""
    return v01 if t == 0 else v01 + quad(f, 0, t)[0]

def desplazamiento_centro(t):
    """x(t) = x0 + ∫₀ᵗ v(s) ds"""
    if t == 0:
        return x01
    tiempos_int = np.linspace(0, t, 100)
    velocidades_int = np.array([velocidad(ti) for ti in tiempos_int])
    return x01 + np.trapz(velocidades_int, tiempos_int)
# CINEMÁTICA
print("\n" + "="*80)
print("CINEMÁTICA: ECUACIÓN ORIGINAL")
print("="*80)

tiempos = np.linspace(0, 36000, 500)
print("Calculando cinemática...")
posiciones = np.array([desplazamiento_centro(t) for t in tiempos])
velocidades = np.array([velocidad(t) for t in tiempos])
aceleraciones = np.array([f(t) for t in tiempos])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11))

ax1.plot(tiempos, posiciones, 'b-', linewidth=2.5, label='Posición x(t)')
ax1.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Posición (km)', fontsize=12, fontweight='bold')
ax1.set_title('POSICIÓN: x(t) = x₀ + ∫₀ᵗ v(s) ds', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper left')

ax2.plot(tiempos, velocidades, 'r-', linewidth=2.5, label='Velocidad v(t)')
ax2.axhline(y=np.max(velocidades), color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Velocidad (km/s)', fontsize=12, fontweight='bold')
ax2.set_title('VELOCIDAD: v(t) = v₀ + ∫₀ᵗ a(s) ds', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=10, loc='upper left')

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
print(f"  Velocidad final:          {velocidades[-1]:.2f} km/s")
print(f"  Aceleración inicial:      {aceleraciones[1]:.4f} km/s²")
print(f"  Aceleración máxima:       {np.max(aceleraciones):.4f} km/s²")
print(f"  Posición inicial:         {x01:.2f} km")
print(f"  Posición final:           {posiciones[-1]:.2f} km")
print(f"  Distancia recorrida:      {posiciones[-1] - x01:.2f} km")


# VISUALIZACIÓN: CME EN COORDENADAS POLARES
print("\n" + "="*80)
print("VISUALIZACIÓN: PROPAGACIÓN Y EVOLUCIÓN DE CME")
print("="*80)

ESTADOS_TIEMPO = 10
tiempos_frames = np.linspace(0, 36000, ESTADOS_TIEMPO)
r_cme_max = R_CME_INICIAL + 0.08 * tiempos_frames[-1]
LIMITE_RADIO_MAX = r_cme_max * 3

theta = np.linspace(0, 2*np.pi, 360)
r = np.linspace(0, LIMITE_RADIO_MAX, 100)
THETA, R = np.meshgrid(theta, r)

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Propagación de CME: Expansión de Círculos Concéntricos (Coordenadas Polares)', 
             fontsize=18, fontweight='bold', y=0.99)

for idx, t_frame in enumerate(tiempos_frames):
    print(f"  Frame {idx+1}/{ESTADOS_TIEMPO}: t = {t_frame:.0f} s ({t_frame/3600:.1f} h)", end="... ")
    
    v_t = velocidad(t_frame)
    r_cme = R_CME_INICIAL + (v_t / 100.0) * (t_frame / 100.0)
    
    ax = plt.subplot(2, 5, idx + 1, projection='polar')

    
    # RADIO Y DENSIDAD - FORMA DE MEDIA LUNA (DESPRENDIDA DEL SOL)
    # Radio exterior deformado
    r_cme_exterior = r_cme * (0.7 + 0.5 * np.cos(THETA))
    # Radio interior: CME se despega del Sol
    r_cme_interior = r_cme * (0.3 + 0.15 * np.cos(THETA))
    # CME es un anillo entre interior y exterior
    mascara_cme = (R > r_cme_interior) & (R <= r_cme_exterior)
    
    # Distribución angular (máxima en frente θ=0): más contraste
    dens_angular = (1.0 + np.cos(THETA))**5
    
    # Concentración en frente: gaussiana en borde radial r ≈ r_cme
    r_norm = R / (r_cme + 0.1)
    dens_radial = np.exp(-8.0 * (r_norm - 1.0)**2)
    
    # Factores de dilución: expansión (∝ 1/√r) y tiempo (∝ 1/√t)
    expansion_factor = (r_cme / R_CME_INICIAL)**0.5
    t_norm = np.maximum(1.0, t_frame / 600.0)
    time_factor = 1.0 / np.sqrt(t_norm)
    
    densidad_diluida = 100.0 / expansion_factor * time_factor
    dens_cme = DENSIDAD_FONDO * densidad_diluida * dens_angular * (0.3 + dens_radial)
    
    dens_plot = np.where(mascara_cme, dens_cme, DENSIDAD_FONDO)
    dens_plot = np.clip(dens_plot, DENSIDAD_FONDO, np.nanmax(dens_plot))
    dens_plot_log = np.log10(np.maximum(dens_plot, DENSIDAD_FONDO/10.0))
    
    dens_min, dens_max = float(np.nanmin(dens_plot_log)), float(np.nanmax(dens_plot_log))
    if dens_max <= dens_min:
        dens_max = dens_min + 0.1
    
    levels_dens = np.linspace(dens_min, dens_max, 100)
    ax.contourf(THETA, R, dens_plot_log, levels=levels_dens, cmap='turbo', alpha=0.9)
    
    # CAMPO DE VELOCIDADES
    theta_vec = np.linspace(0, 2*np.pi, 22)
    r_vec = np.linspace(0.5, LIMITE_RADIO_MAX, 6)
    THETA_vec, R_vec = np.meshgrid(theta_vec, r_vec)
    # Máscara con deformación de media luna (CME desprendida)
    r_cme_deformado_ext = r_cme * (0.7 + 0.5 * np.cos(THETA_vec))
    r_cme_deformado_int = r_cme * (0.3 + 0.15 * np.cos(THETA_vec))
    mascara_vec = (R_vec > r_cme_deformado_int) & (R_vec <= r_cme_deformado_ext)
    
    v_t = velocidad(t_frame)
    factor_dens = np.clip((1.0 + np.cos(THETA_vec))**2.5, 0, 1)
    
    v_radial = v_t * factor_dens * (1.0 + 0.3 * np.cos(THETA_vec)**2)
    v_tangencial = 0.05 * v_t * np.sin(2*THETA_vec) * factor_dens
    
    v_mag = np.sqrt(v_radial**2 + v_tangencial**2)
    v_mag[v_mag == 0] = 1
    v_radial_norm = v_radial / v_mag
    v_tangencial_norm = v_tangencial / v_mag
    
    v_rad_m = np.where(mascara_vec, v_radial_norm * factor_dens, np.nan)
    v_tang_m = np.where(mascara_vec, v_tangencial_norm * factor_dens, np.nan)
    
    U = v_rad_m * np.cos(THETA_vec) - v_tang_m * np.sin(THETA_vec)
    V = v_rad_m * np.sin(THETA_vec) + v_tang_m * np.cos(THETA_vec)
    
    ax.quiver(THETA_vec, R_vec, U, V, scale=20, width=0.004, color='lime', alpha=0.9)
    
    # ETIQUETADO
    titulo = f"t = {t_frame/60:.1f} min"
    ax.set_title(titulo, fontsize=10, fontweight='bold', pad=10)
    ax.set_ylim([0, LIMITE_RADIO_MAX])
    ax.set_rlabel_position(45)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    print("✓")

plt.tight_layout(rect=[0, 0, 0.92, 0.97])
cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.75])
sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=dens_min, vmax=dens_max))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('log₁₀(ρ) [protones/cm³]', rotation=270, labelpad=25, fontsize=11, fontweight='bold')

plt.savefig("cme_evolucion_polar.pdf", dpi=300, bbox_inches='tight')
print("\n✓ Visualización guardada: cme_evolucion_polar.pdf")
plt.show()

# ANÁLISIS FINAL
print("\n" + "="*80)
print("ANÁLISIS: DENSIDAD-VELOCIDAD")
print("="*80)

t_analisis = 1500
v_a = velocidad(t_analisis)
a_a = f(t_analisis)

print(f"\nEn t = {t_analisis} s:")
print(f"  Velocidad: {v_a:.2f} km/s")
print(f"  Aceleración: {a_a:.4f} km/s²")
print(f"  La densidad máxima está en el FRENTE (θ ≈ 0)")
print(f"  donde la velocidad radial es máxima.")

print("\n" + "="*80)
print("✓ SIMULACIÓN COMPLETADA")
print("="*80)
