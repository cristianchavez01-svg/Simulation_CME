import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import matplotlib

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DE FUENTES
# ──────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.titleweight'] = 'normal'

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
R_SOL_KM       = 695700
R_SOL_STR      = r'$R_\odot$'
DENSIDAD_FONDO = 100
T_HORAS        = 4
FACTOR_ESCALA  = R_SOL_KM

# ──────────────────────────────────────────────────────────────────────────────
# PARÁMETROS FÍSICOS CME-1
# ──────────────────────────────────────────────────────────────────────────────
# PARÁMETROS FÍSICOS
tr1, td1 = 138, 1249          # Tiempos característicos (s)
ar1, ad1 = 0.001, 4.950        # Amplitudes (km/s²)
v01, x01 = 40, 25000          # Velocidad (km/s) y posición inicial (km)
R_CME_INICIAL = 2.0           # Radio inicial de CME
SEMILLA     = 436
COLOR_CINE  = 'steelblue'

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES CINEMÁTICAS
# ──────────────────────────────────────────────────────────────────────────────
def aceleracion(s):
    return (ar1 * ad1) / (ad1 * np.exp(-s / tr1) + ar1 * np.exp(s / td1))

def velocidad(t):
    if t <= 0:
        return v01
    return v01 + quad(aceleracion, 0, t)[0]

def posicion(t):
    if t == 0:
        return x01
    tiempos_int = np.linspace(0, t, 100)
    vels = np.array([v01 + (quad(aceleracion, 0, ti)[0] if ti > 0 else 0)
                     for ti in tiempos_int])
    return x01 + np.trapz(vels, tiempos_int)

# ──────────────────────────────────────────────────────────────────────────────
# MORFOLOGÍA (coordenadas polares)
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=SEMILLA)
N_MODOS    = 7
amp_ext    = rng.uniform(0.03, 0.08, N_MODOS)
fase_ext   = rng.uniform(0, 2*np.pi, N_MODOS)
amp_gros   = rng.uniform(0.02, 0.07, N_MODOS)
fase_gros  = rng.uniform(0, 2*np.pi, N_MODOS)
asimetria  = rng.uniform(0.85, 1.15)
N_FIL      = rng.integers(2, 5)
ang_fil    = rng.uniform(-np.pi/2, np.pi/2, N_FIL)
amp_fil    = rng.uniform(0.10, 0.20, N_FIL)
ancho_fil  = rng.uniform(0.30, 0.45, N_FIL)

def _ruido_fourier(theta_arr, amps, fases):
    ruido = np.zeros_like(theta_arr)
    for k, (a, ph) in enumerate(zip(amps, fases), start=2):
        ruido += a * np.cos(k * theta_arr + ph)
    return ruido

def _filamentos(theta_arr):
    fil = np.zeros_like(theta_arr)
    for ang, amp, ancho in zip(ang_fil, amp_fil, ancho_fil):
        fil += amp * np.exp(-((theta_arr - ang)**2) / (2 * ancho**2))
    return fil

def forma(theta_arr, r_cme):
    apertura   = np.pi * 0.56
    ventana    = np.clip(np.cos(theta_arr / apertura * (np.pi/2)), 0, 1)**2
    theta_asim = theta_arr * np.where(theta_arr >= 0, asimetria, 2 - asimetria)
    r_ext_base  = r_cme * (0.75 + 0.45 * np.cos(theta_asim))
    r_ext_ruido = r_cme * _ruido_fourier(theta_arr, amp_ext, fase_ext)
    r_ext_fil   = r_cme * _filamentos(theta_arr)
    r_exterior  = (r_ext_base + r_ext_ruido + r_ext_fil) * ventana
    grosor_base  = 0.28 + 0.10 * np.cos(theta_asim)
    grosor_ruido = _ruido_fourier(theta_arr, amp_gros, fase_gros)
    grosor       = np.clip(grosor_base + grosor_ruido, 0.10, 0.50)
    r_interior   = r_exterior * (1.0 - grosor) * ventana
    r_interior   = np.maximum(r_interior, r_cme * 0.15 * ventana)
    return r_exterior, r_interior

def densidad_campo(THETA, R, r_cme, t_frame):
    r_ext_2d, r_int_2d = forma(THETA, r_cme)
    mascara = (R > r_int_2d) & (R <= r_ext_2d)
    dens_angular     = np.clip((1.0 + np.cos(THETA))**5, 0, None)
    r_norm           = R / (r_cme + 0.1)
    dens_radial      = np.exp(-8.0 * (r_norm - 1.0)**2)
    expansion_factor = (r_cme / R_CME_INICIAL)**0.5
    t_norm           = np.maximum(1.0, t_frame / 600.0)
    time_factor      = 1.0 / np.sqrt(t_norm)
    densidad_diluida = 100.0 / expansion_factor * time_factor
    dens_cme = DENSIDAD_FONDO * densidad_diluida * dens_angular * (0.3 + dens_radial)
    campo = np.where(mascara, dens_cme, np.nan)
    campo = np.where(mascara, np.clip(campo, DENSIDAD_FONDO,
                                      np.nanmax(campo) if np.any(mascara) else DENSIDAD_FONDO),
                     np.nan)
    return campo, mascara

def campo_velocidad(THETA_vec, R_vec, r_cme, v_t):
    r_ext_vec, r_int_vec = forma(THETA_vec, r_cme)
    mascara_vec  = (R_vec > r_int_vec) & (R_vec <= r_ext_vec)
    factor_dens  = np.clip((1.0 + np.cos(THETA_vec))**2.5, 0, 1)
    v_radial     = v_t * factor_dens * (1.0 + 0.3 * np.cos(THETA_vec)**2)
    v_tangencial = 0.05 * v_t * np.sin(2*THETA_vec) * factor_dens
    v_mag = np.sqrt(v_radial**2 + v_tangencial**2)
    v_mag[v_mag == 0] = 1
    v_rad_n  = v_radial  / v_mag
    v_tang_n = v_tangencial / v_mag
    v_rad_m  = np.where(mascara_vec, v_rad_n  * factor_dens, np.nan)
    v_tang_m = np.where(mascara_vec, v_tang_n * factor_dens, np.nan)
    U = v_rad_m * np.cos(THETA_vec) - v_tang_m * np.sin(THETA_vec)
    V = v_rad_m * np.sin(THETA_vec) + v_tang_m * np.cos(THETA_vec)
    return U, V, mascara_vec

# ──────────────────────────────────────────────────────────────────────────────
# 1. CINEMÁTICA
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("CINEMÁTICA: CME-1")
print("="*80)

tiempos   = np.linspace(0, T_HORAS * 3600, 500)
tiempos_h = tiempos / 3600

print("Calculando cinemática...")
posiciones    = np.array([posicion(t) for t in tiempos], dtype=float)
velocidades   = np.array([velocidad(t) for t in tiempos], dtype=float)
aceleraciones = np.array([aceleracion(t) for t in tiempos], dtype=float) * 1000.0  # km/s² → m/s²

posiciones_rs = posiciones / R_SOL_KM

# Etapas
idx_amax = np.argmax(aceleraciones)
t_inic   = tiempos_h[idx_amax]
v_umbral = 0.95 * np.max(velocidades)
t_acel   = tiempos_h[np.argmax(velocidades >= v_umbral)]

print(f"  Fin iniciación  (a_max):      t = {t_inic:.2f} h")
print(f"  Fin aceleración (95% v_max):  t = {t_acel:.2f} h")

# Colores de fondo en escala de grises
C_BLANCO = '#FFFFFF'
C_GRIS   = '#CCCCCC'

def sombrear_etapas(ax):
    ax.axvspan(0,      t_inic,  color=C_BLANCO, alpha=1.0, zorder=0)
    ax.axvspan(t_inic, t_acel,  color=C_GRIS,   alpha=0.5, zorder=0)
    ax.axvspan(t_acel, T_HORAS, color=C_BLANCO, alpha=1.0, zorder=0)
    ax.axvline(x=t_inic, color='k', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(x=t_acel, color='k', linestyle=':',  linewidth=0.8, alpha=0.5, zorder=2)

def etiquetas_etapas(ax):
    ylim  = ax.get_ylim()
    rango = ylim[1] - ylim[0]
    ypos  = ylim[0] + rango * 0.75

    for t_mid, label in [((0 + t_inic)/2,      'Iniciación'),
                          ((t_inic + t_acel)/2, 'Aceleración')]:
        ax.text(t_mid, ypos, label, ha='center', fontsize=11,
                color='#444444', zorder=4, rotation=90, va='center')

    ax.text((t_acel + T_HORAS)/2, ypos, 'Propagación', ha='center',
            fontsize=11, color='#444444', zorder=4, rotation=0, va='center')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11),
                                     sharex=True, gridspec_kw={'hspace': 0})
fig.suptitle('Cinemática CME de prueba', fontsize=20, y=0.96)
fig.text(0.5, 0.90, f'{T_HORAS} horas de propagación',
         ha='center', fontsize=13, style='italic', color='#444444')

sombrear_etapas(ax1)
ax1.plot(tiempos_h, posiciones_rs, color=COLOR_CINE, linewidth=2.5, zorder=3)
ax1.set_ylabel(f'Posición ({R_SOL_STR})', fontsize=16)
ax1.set_xlim(0, T_HORAS)
ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax1.tick_params(bottom=False)
etiquetas_etapas(ax1)

sombrear_etapas(ax2)
ax2.plot(tiempos_h, velocidades, color=COLOR_CINE, linewidth=2.5, zorder=3)
ax2.set_ylabel('Velocidad (km/s)', fontsize=16)
ax2.set_xlim(0, T_HORAS)
ax2.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax2.tick_params(bottom=False)

sombrear_etapas(ax3)
ax3.plot(tiempos_h, aceleraciones, color=COLOR_CINE, linewidth=2.5, zorder=3)
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5, zorder=2)
ax3.set_xlabel('Tiempo (h)', fontsize=16)
ax3.set_ylabel(r'Aceleración (m/s$^2$)', fontsize=16)
ax3.set_xlim(0, T_HORAS)
ax3.set_xticks(np.arange(0, T_HORAS + 1, 1))
ax3.grid(True, alpha=0.3, linestyle='--', zorder=1)

plt.savefig("cinematica_original_prueba.pdf", dpi=300, bbox_inches='tight')
print("✓ Gráfica cinemática guardada: cinematica_original_prueba.pdf")
plt.show()

print("\nRESUMEN CINEMÁTICO:")
print(f"  Velocidad inicial:    {v01:.2f} km/s")
print(f"  Velocidad máxima:     {np.max(velocidades):.2f} km/s")
print(f"  Velocidad final:      {velocidades[-1]:.2f} km/s")
print(f"  Aceleración máxima:   {np.max(aceleraciones):.4f} m/s²")
print(f"  Posición inicial:     {x01 / R_SOL_KM:.4f} R_sol")
print(f"  Posición final:       {posiciones[-1] / R_SOL_KM:.4f} R_sol")
print(f"  Distancia recorrida:  {(posiciones[-1] - x01) / R_SOL_KM:.4f} R_sol")

# ──────────────────────────────────────────────────────────────────────────────
# 2. PROPAGACIÓN EN COORDENADAS POLARES
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("VISUALIZACIÓN POLAR: CME-1")
print("="*80)

ESTADOS_TIEMPO = 8
ETIQUETAS      = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
tiempos_frames = np.linspace(3600, T_HORAS * 3600, ESTADOS_TIEMPO)

pos_final        = posicion(tiempos_frames[-1]) / FACTOR_ESCALA
LIMITE_RADIO_MAX = pos_final * 1.4

# Malla polar global — alta resolución
theta    = np.linspace(-np.pi, np.pi, 800)
r        = np.linspace(0, LIMITE_RADIO_MAX, 400)
THETA, R = np.meshgrid(theta, r)

# Malla de vectores global
theta_vec        = np.linspace(-np.pi, np.pi, 30)
r_vec            = np.linspace(0.5, LIMITE_RADIO_MAX, 8)
THETA_vec, R_vec = np.meshgrid(theta_vec, r_vec)

# Pre-cálculo del máximo global de densidad
print("  Pre-calculando máximo global de densidad...")
dens_max_global = DENSIDAD_FONDO
for t_pre in tiempos_frames:
    r_pre = posicion(t_pre) / FACTOR_ESCALA
    c_pre, m_pre = densidad_campo(THETA, R, r_pre, t_pre)
    frame_max = float(np.nanmax(c_pre)) if np.any(~np.isnan(c_pre)) else DENSIDAD_FONDO
    if frame_max > dens_max_global:
        dens_max_global = frame_max

DENS_MIN_GLOBAL = float(np.log10(DENSIDAD_FONDO))
DENS_MAX_GLOBAL = max(float(np.log10(dens_max_global)), DENS_MIN_GLOBAL + 0.1)
print(f"  Rango de densidad: [{DENS_MIN_GLOBAL:.3f}, {DENS_MAX_GLOBAL:.3f}] log10")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Propagación de CME - 1', fontsize=18, fontweight='normal', y=0.99)
fig.text(0.5, 0.94, f'{T_HORAS} horas de propagación',
         ha='center', fontsize=13, style='italic', color='#444444')

for idx, t_frame in enumerate(tiempos_frames):
    print(f"  Frame {idx+1}/{ESTADOS_TIEMPO}: t = {t_frame/3600:.1f} h", end=" ... ")

    ax = plt.subplot(2, 4, idx + 1, projection='polar')
    ax.text(0.05, 0.97, f'{ETIQUETAS[idx]})',
            transform=ax.transAxes, fontsize=12, fontweight='normal',
            va='top', ha='left', color='black', fontstyle='italic')

    r_cme = posicion(t_frame) / FACTOR_ESCALA
    v_t   = velocidad(t_frame)

    # Malla local adaptativa para los primeros 4 frames
    if idx < 4:
        r_max_local = r_cme * 2.7
        r_max_local = max(r_max_local, 1.0)
        th_loc = np.linspace(-np.pi, np.pi, 800)
        r_loc  = np.linspace(0, r_max_local, 400)
        TH_loc, R_loc = np.meshgrid(th_loc, r_loc)
        tv_loc = np.linspace(-np.pi, np.pi, 30)
        rv_loc = np.linspace(0.1, r_max_local, 8)
        THv_loc, Rv_loc = np.meshgrid(tv_loc, rv_loc)
    else:
        TH_loc,  R_loc  = THETA, R
        THv_loc, Rv_loc = THETA_vec, R_vec

    # Fondo azul claro
    ax.contourf(TH_loc, R_loc, np.ones_like(R_loc),
                levels=[0.5, 1.5], colors=['#D6EAF8'], alpha=0.8)

    # Campo de densidad
    campo, mascara = densidad_campo(TH_loc, R_loc, r_cme, t_frame)
    dens_log = np.log10(np.where(~np.isnan(campo),
                                  np.maximum(campo, DENSIDAD_FONDO / 10.0),
                                  np.nan))
    levels_dens = np.linspace(DENS_MIN_GLOBAL, DENS_MAX_GLOBAL, 100)
    ax.contourf(TH_loc, R_loc, dens_log, levels=levels_dens, cmap='viridis', alpha=0.9)

    # Vectores de velocidad
    U, V, mv = campo_velocidad(THv_loc, Rv_loc, r_cme, v_t)
    ax.quiver(THv_loc, Rv_loc, U, V,
              scale=20, width=0.004, color='red', alpha=0.9)

    ax.set_title(f"t = {t_frame/3600:.1f} h  |  r = {r_cme:.2f} {R_SOL_STR}",
                 fontsize=9, fontweight='normal', pad=10)

    if idx < 4:
        ax.set_ylim([0, r_max_local])
    else:
        ax.set_ylim([0, LIMITE_RADIO_MAX])

    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    print(f"r = {r_cme:.2f} R_sol ✓")

plt.tight_layout(rect=[0, 0, 0.92, 0.97])

cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.75])
sm = plt.cm.ScalarMappable(cmap='viridis',
                            norm=plt.Normalize(vmin=DENS_MIN_GLOBAL, vmax=DENS_MAX_GLOBAL))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'log$_{10}$($\rho$) [protones/cm$^3$]',
               rotation=270, labelpad=25, fontsize=11)

plt.savefig("cme_evolucion_polar_prueba.pdf", dpi=300, bbox_inches='tight')
print("\n✓ Visualización guardada: cme_evolucion_polar_prueba.pdf")
plt.show()

print("\n" + "="*80)
print("✓ SIMULACIÓN CME-1 COMPLETADA")
print("="*80)
