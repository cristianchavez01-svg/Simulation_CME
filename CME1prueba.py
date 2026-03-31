import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import matplotlib

# Fuente tipo LaTeX (Computer Modern)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.titleweight'] = 'normal'

# PARÁMETROS FÍSICOS
tr1, td1 = 138, 1249
ar1, ad1 = 0.001, 4.950
v01, x01 = 40, 25000
DENSIDAD_FONDO = 100
R_CME_INICIAL = 2.0
R_SOL_KM  = 695700
R_SOL_STR = r'$R_\odot$'
semilla = 13

def f(s):
    return (ar1 * ad1) / (ad1 * np.exp(-s / tr1) + ar1 * np.exp(s / td1))

def velocidad(t):
    return v01 if t == 0 else v01 + quad(f, 0, t)[0]

def desplazamiento_centro(t):
    if t == 0:
        return x01
    tiempos_int = np.linspace(0, t, 100)
    velocidades_int = np.array([velocidad(ti) for ti in tiempos_int])
    return x01 + np.trapz(velocidades_int, tiempos_int)

# CINEMÁTICA
print("\n" + "="*80)
print("CINEMÁTICA: ECUACIÓN ORIGINAL")
print("="*80)

T_HORAS = 10

tiempos = np.linspace(0, T_HORAS * 3600, 500)
tiempos_h = tiempos / 3600

print("Calculando cinemática...")
posiciones    = np.array([desplazamiento_centro(t) for t in tiempos])
velocidades   = np.array([velocidad(t) for t in tiempos])
aceleraciones = np.array([f(t) for t in tiempos])

posiciones_rs = posiciones / R_SOL_KM

idx_amax = np.argmax(aceleraciones)
t_inic   = tiempos_h[idx_amax]

v_umbral  = 0.95 * np.max(velocidades)
idx_prop  = np.argmax(velocidades >= v_umbral)
t_acel    = tiempos_h[idx_prop]

print(f"  Fin iniciación  (a_max):        t = {t_inic:.2f} h")
print(f"  Fin aceleración (95% v_max):    t = {t_acel:.2f} h")

C_INIC = '#FFFFFF'
C_ACEL = '#CCCCCC'
C_PROP = '#FFFFFF'

def agregar_etapas(ax):
    ax.axvspan(0,       t_inic,  color=C_INIC, alpha=1.0, zorder=0)
    ax.axvspan(t_inic,  t_acel,  color=C_ACEL, alpha=0.6, zorder=0)
    ax.axvspan(t_acel,  T_HORAS, color=C_PROP, alpha=1.0, zorder=0)
    ax.axvline(x=t_inic, color='k', linestyle='--', linewidth=1.0, alpha=0.6, zorder=2)
    ax.axvline(x=t_acel, color='k', linestyle=':',  linewidth=1.2, alpha=0.6, zorder=2)

def etiquetas_etapas(ax):
    ylim = ax.get_ylim()
    rango = ylim[1] - ylim[0]
    ypos_alto = ylim[0] + rango * 0.92
    ypos_bajo = ylim[0] + rango * 0.5
    ax.text((0 + t_inic) / 2,       ypos_bajo, 'Iniciación',  ha='center', fontsize=15, color='#333333', zorder=4, rotation=90)
    ax.text((t_inic + t_acel) / 2,  ypos_bajo, 'Aceleración', ha='center', fontsize=15, color='#333333', zorder=4, rotation=90)
    ax.text((t_acel + T_HORAS) / 2, ypos_alto, 'Propagación', ha='center', fontsize=15, color='#333333', zorder=4)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11),
                                     sharex=True,
                                     gridspec_kw={'hspace': 0})

fig.suptitle('Cinemática de la CME - 1', fontsize=20, y=0.98)
fig.text(0.5, 0.935, f'{T_HORAS} horas de propagación',
         ha='center', fontsize=13, style='italic', color='#444444')

agregar_etapas(ax1)
ax1.plot(tiempos_h, posiciones_rs, 'b-', linewidth=2.5, zorder=3)
ax1.set_ylabel(f'Posición ({R_SOL_STR})', fontsize=18)
ax1.set_xlim(0, T_HORAS)
ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax1.tick_params(bottom=False)
etiquetas_etapas(ax1)

agregar_etapas(ax2)
ax2.plot(tiempos_h, velocidades, 'r-', linewidth=2.5, zorder=3)
ax2.set_ylabel('Velocidad (km/s)', fontsize=18)
ax2.set_xlim(0, T_HORAS)
ax2.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax2.tick_params(bottom=False)

agregar_etapas(ax3)
ax3.plot(tiempos_h, aceleraciones, 'g-', linewidth=2.5, zorder=3)
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5, zorder=2)
ax3.set_xlabel('Tiempo (h)', fontsize=18)
ax3.set_ylabel(f'Aceleración (km/s$^2$)', fontsize=18)
ax3.set_xlim(0, T_HORAS)
ax3.set_xticks(np.arange(0, T_HORAS + 1, 1))
ax3.grid(True, alpha=0.3, linestyle='--', zorder=1)

plt.savefig("cinematica_original.pdf", dpi=300, bbox_inches='tight')
print("✓ Gráfica cinemática guardada: cinematica_original.pdf")
plt.show()

print("\nRESUMEN CINEMÁTICO:")
print(f"  Velocidad inicial:        {v01:.2f} km/s")
print(f"  Velocidad máxima:         {np.max(velocidades):.2f} km/s")
print(f"  Velocidad final:          {velocidades[-1]:.2f} km/s")
print(f"  Aceleración inicial:      {aceleraciones[1]:.4f} km/s²")
print(f"  Aceleración máxima:       {np.max(aceleraciones):.4f} km/s²")
print(f"  Posición inicial:         {x01 / R_SOL_KM:.4f} R_sol")
print(f"  Posición final:           {posiciones[-1] / R_SOL_KM:.4f} R_sol")
print(f"  Distancia recorrida:      {(posiciones[-1] - x01) / R_SOL_KM:.4f} R_sol")


# VISUALIZACIÓN: CME EN COORDENADAS POLARES
print("\n" + "="*80)
print("VISUALIZACIÓN: PROPAGACIÓN Y EVOLUCIÓN DE CME")
print("="*80)

FACTOR_ESCALA = R_SOL_KM
ETIQUETAS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

rng = np.random.default_rng(seed=semilla)

N_MODOS = 6
amp_ext   = rng.uniform(0.03, 0.08, N_MODOS)
fase_ext  = rng.uniform(0, 2*np.pi, N_MODOS)
amp_gros  = rng.uniform(0.02, 0.07, N_MODOS)
fase_gros = rng.uniform(0, 2*np.pi, N_MODOS)
asimetria = rng.uniform(0.85, 1.15)
N_FILAMENTOS = rng.integers(2, 5)
ang_fil   = rng.uniform(-np.pi/2, np.pi/2, N_FILAMENTOS)
amp_fil   = rng.uniform(0.10, 0.20, N_FILAMENTOS)
ancho_fil = rng.uniform(0.30, 0.45, N_FILAMENTOS)

def ruido_fourier(theta_arr, amplitudes, fases):
    ruido = np.zeros_like(theta_arr)
    for k, (a, ph) in enumerate(zip(amplitudes, fases), start=2):
        ruido += a * np.cos(k * theta_arr + ph)
    return ruido

def filamentos(theta_arr):
    fil = np.zeros_like(theta_arr)
    for ang, amp, ancho in zip(ang_fil, amp_fil, ancho_fil):
        fil += amp * np.exp(-((theta_arr - ang)**2) / (2 * ancho**2))
    return fil

def forma_cme(theta_arr, r_cme):
    apertura = np.pi * 0.56
    ventana  = np.clip(np.cos(theta_arr / apertura * (np.pi/2)), 0, 1)**2
    theta_asim  = theta_arr * np.where(theta_arr >= 0, asimetria, 2 - asimetria)
    r_ext_base  = r_cme * (0.75 + 0.45 * np.cos(theta_asim))
    r_ext_ruido = r_cme * ruido_fourier(theta_arr, amp_ext, fase_ext)
    r_ext_fil   = r_cme * filamentos(theta_arr)
    r_exterior  = (r_ext_base + r_ext_ruido + r_ext_fil) * ventana
    grosor_base  = 0.28 + 0.10 * np.cos(theta_asim)
    grosor_ruido = ruido_fourier(theta_arr, amp_gros, fase_gros)
    grosor       = np.clip(grosor_base + grosor_ruido, 0.10, 0.50)
    r_interior   = r_exterior * (1.0 - grosor) * ventana
    r_interior   = np.maximum(r_interior, r_cme * 0.15 * ventana)
    return r_exterior, r_interior

ESTADOS_TIEMPO   = 8
tiempos_frames   = np.linspace(3600, 28800, ESTADOS_TIEMPO)

# Radio máximo global (usado por los últimos 4 frames)
pos_final        = desplazamiento_centro(tiempos_frames[-1]) / FACTOR_ESCALA
LIMITE_RADIO_MAX = pos_final * 1.4

# Malla global (la más grande posible para cubrir todos los frames)
theta = np.linspace(-np.pi, np.pi, 360)
r     = np.linspace(0, LIMITE_RADIO_MAX, 100)
THETA, R = np.meshgrid(theta, r)

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Propagación de CME - 1',
             fontsize=18, fontweight='normal', y=0.99)
fig.text(0.5, 0.935, f'{T_HORAS} horas de propagación',
         ha='center', fontsize=13, style='italic', color='#444444')


for idx, t_frame in enumerate(tiempos_frames):
    print(f"  Frame {idx+1}/{ESTADOS_TIEMPO}: t = {t_frame:.0f} s ({t_frame/3600:.1f} h)", end="... ")

    v_t   = velocidad(t_frame)
    r_cme = desplazamiento_centro(t_frame) / FACTOR_ESCALA

    ax = plt.subplot(2, 4, idx + 1, projection='polar')

    # Etiqueta estilo LaTeX simple: a), b), c)...
    ax.text(0.05, 0.97, f'{ETIQUETAS[idx]})',
            transform=ax.transAxes,
            fontsize=12, fontweight='normal',
            va='top', ha='left', color='black',
            fontstyle='italic')

    r_ext_2d, r_int_2d = forma_cme(THETA, r_cme)
    mascara_cme   = (R > r_int_2d) & (R <= r_ext_2d)
    mascara_fondo = ~mascara_cme

    dens_angular = np.clip((1.0 + np.cos(THETA))**5, 0, None)
    r_norm       = R / (r_cme + 0.1)
    dens_radial  = np.exp(-8.0 * (r_norm - 1.0)**2)

    expansion_factor = (r_cme / R_CME_INICIAL)**0.5
    t_norm       = np.maximum(1.0, t_frame / 600.0)
    time_factor  = 1.0 / np.sqrt(t_norm)

    densidad_diluida = 100.0 / expansion_factor * time_factor
    dens_cme = DENSIDAD_FONDO * densidad_diluida * dens_angular * (0.3 + dens_radial)

    dens_plot     = np.where(mascara_cme, dens_cme, np.nan)
    dens_plot     = np.clip(dens_plot, DENSIDAD_FONDO, np.nanmax(dens_plot))
    dens_plot_log = np.log10(np.maximum(dens_plot, DENSIDAD_FONDO / 10.0))

    dens_min = float(np.log10(DENSIDAD_FONDO))
    dens_max = float(np.nanmax(dens_plot_log))
    if dens_max <= dens_min:
        dens_max = dens_min + 0.1

    ax.contourf(THETA, R,
                np.where(mascara_fondo, 1.0, np.nan),
                levels=[0.5, 1.5], colors=['#D6EAF8'], alpha=0.8)

    levels_dens = np.linspace(dens_min, dens_max, 100)
    ax.contourf(THETA, R, dens_plot_log, levels=levels_dens, cmap='viridis', alpha=0.9)

    theta_vec = np.linspace(-np.pi, np.pi, 22)
    r_vec     = np.linspace(0.5, LIMITE_RADIO_MAX, 6)
    THETA_vec, R_vec = np.meshgrid(theta_vec, r_vec)

    r_ext_vec, r_int_vec = forma_cme(THETA_vec, r_cme)
    mascara_vec = (R_vec > r_int_vec) & (R_vec <= r_ext_vec)

    factor_dens       = np.clip((1.0 + np.cos(THETA_vec))**2.5, 0, 1)
    v_radial          = v_t * factor_dens * (1.0 + 0.3 * np.cos(THETA_vec)**2)
    v_tangencial      = 0.05 * v_t * np.sin(2*THETA_vec) * factor_dens

    v_mag = np.sqrt(v_radial**2 + v_tangencial**2)
    v_mag[v_mag == 0] = 1
    v_radial_norm     = v_radial / v_mag
    v_tangencial_norm = v_tangencial / v_mag

    v_rad_m  = np.where(mascara_vec, v_radial_norm * factor_dens, np.nan)
    v_tang_m = np.where(mascara_vec, v_tangencial_norm * factor_dens, np.nan)

    U = v_rad_m * np.cos(THETA_vec) - v_tang_m * np.sin(THETA_vec)
    V = v_rad_m * np.sin(THETA_vec) + v_tang_m * np.cos(THETA_vec)

    ax.quiver(THETA_vec, R_vec, U, V, scale=20, width=0.004, color='red', alpha=0.9)

    titulo = f"t = {t_frame/3600:.1f} h  |  r = {r_cme:.2f} {R_SOL_STR}"
    ax.set_title(titulo, fontsize=9, fontweight='normal', pad=10)

    # Primeros 4 frames: ylim proporcional al r_cme (zoom progresivo)
    # Últimos 4 frames: ylim fijo al máximo global
    if idx < 4:
        limite_frame = r_cme * 2.7   # margen de 2x para mostrar la CME completa
        ax.set_ylim([0, limite_frame])
    else:
        ax.set_ylim([0, LIMITE_RADIO_MAX])

    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

    print(f"r = {r_cme:.2f} R_sol ✓")

plt.tight_layout(rect=[0, 0, 0.92, 0.97])
cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.75])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=dens_min, vmax=dens_max))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(f'log$_{{10}}$($\\rho$) [protones/cm$^3$]', rotation=270, labelpad=25,
               fontsize=11, fontweight='normal')

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
print(f"  Velocidad:     {v_a:.2f} km/s")
print(f"  Aceleración:   {a_a:.4f} km/s²")
print(f"  Posición CME:  {desplazamiento_centro(t_analisis) / R_SOL_KM:.4f} R_sol")
print(f"  La densidad máxima está en el FRENTE (θ ≈ 0)")
print(f"  donde la velocidad radial es máxima.")

print("\n" + "="*80)
print("✓ SIMULACIÓN COMPLETADA")
print("="*80)