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
# CONSTANTES COMPARTIDAS
# ──────────────────────────────────────────────────────────────────────────────
R_SOL_KM  = 695700
R_SOL_STR = r'$R_\odot$'
DENSIDAD_FONDO = 100
T_HORAS   = 10
FACTOR_ESCALA = R_SOL_KM

# ──────────────────────────────────────────────────────────────────────────────
# CLASE CME: encapsula parámetros y funciones cinemáticas de cada evento
# ──────────────────────────────────────────────────────────────────────────────
class CME:
    """
    Encapsula los parámetros físicos y las funciones cinemáticas de una CME.
    Acepta un 'retraso' en segundos: la CME no existe antes de t_inicio.
    """

    def __init__(self, nombre, tr, td, ar, ad, v0, x0,
                 R_CME_INICIAL, semilla, color_cine,
                 t_inicio_s=0.0):
        self.nombre       = nombre
        self.tr           = tr
        self.td           = td
        self.ar           = ar
        self.ad           = ad
        self.v0           = v0
        self.x0           = x0
        self.R_CME_INICIAL = R_CME_INICIAL
        self.semilla      = semilla
        self.color_cine   = color_cine   # color para gráficas cinemáticas
        self.t_inicio_s   = t_inicio_s   # retraso de lanzamiento en segundos

        # Genera forma aleatoria reproducible
        rng = np.random.default_rng(seed=semilla)
        N_MODOS = 7
        self.amp_ext   = rng.uniform(0.03, 0.08, N_MODOS)
        self.fase_ext  = rng.uniform(0, 2*np.pi, N_MODOS)
        self.amp_gros  = rng.uniform(0.02, 0.07, N_MODOS)
        self.fase_gros = rng.uniform(0, 2*np.pi, N_MODOS)
        self.asimetria = rng.uniform(0.85, 1.15)
        N_FIL = rng.integers(2, 5)
        self.ang_fil   = rng.uniform(-np.pi/2, np.pi/2, N_FIL)
        self.amp_fil   = rng.uniform(0.10, 0.20, N_FIL)
        self.ancho_fil = rng.uniform(0.30, 0.45, N_FIL)

    # ── Cinemática ────────────────────────────────────────────────────────────

    def aceleracion(self, s):
        """Aceleración como función del tiempo PROPIO de la CME (s >= 0)."""
        return (self.ar * self.ad) / (
            self.ad * np.exp(-s / self.tr) + self.ar * np.exp(s / self.td)
        )

    def velocidad(self, t):
        """Velocidad en tiempo GLOBAL t (segundos)."""
        s = t - self.t_inicio_s          # tiempo propio
        if s <= 0:
            return 0.0                   # CME aún no existe
        return self.v0 + quad(self.aceleracion, 0, s)[0]

    def posicion(self, t):
        """Posición del centro en tiempo GLOBAL t (km)."""
        s = t - self.t_inicio_s
        if s < 0:
            return None                  # CME aún no existe
        if s == 0:
            return self.x0
        tiempos_int = np.linspace(0, s, 100)
        vels        = np.array([self.v0 + (quad(self.aceleracion, 0, ti)[0] if ti > 0 else 0)
                                for ti in tiempos_int])
        return self.x0 + np.trapz(vels, tiempos_int)

    # ── Morfología (coordenadas polares) ─────────────────────────────────────

    def _ruido_fourier(self, theta_arr):
        ruido = np.zeros_like(theta_arr)
        for k, (a, ph) in enumerate(zip(self.amp_ext, self.fase_ext), start=2):
            ruido += a * np.cos(k * theta_arr + ph)
        return ruido

    def _ruido_fourier_gros(self, theta_arr):
        ruido = np.zeros_like(theta_arr)
        for k, (a, ph) in enumerate(zip(self.amp_gros, self.fase_gros), start=2):
            ruido += a * np.cos(k * theta_arr + ph)
        return ruido

    def _filamentos(self, theta_arr):
        fil = np.zeros_like(theta_arr)
        for ang, amp, ancho in zip(self.ang_fil, self.amp_fil, self.ancho_fil):
            fil += amp * np.exp(-((theta_arr - ang)**2) / (2 * ancho**2))
        return fil

    def forma(self, theta_arr, r_cme):
        """
        Devuelve (r_exterior, r_interior) para la morfología de la CME
        centrada en r_cme (en unidades de R_sol).
        """
        apertura   = np.pi * 0.56
        ventana    = np.clip(np.cos(theta_arr / apertura * (np.pi/2)), 0, 1)**2
        theta_asim = theta_arr * np.where(theta_arr >= 0,
                                          self.asimetria, 2 - self.asimetria)
        r_ext_base  = r_cme * (0.75 + 0.45 * np.cos(theta_asim))
        r_ext_ruido = r_cme * self._ruido_fourier(theta_arr)
        r_ext_fil   = r_cme * self._filamentos(theta_arr)
        r_exterior  = (r_ext_base + r_ext_ruido + r_ext_fil) * ventana

        grosor_base  = 0.28 + 0.10 * np.cos(theta_asim)
        grosor_ruido = self._ruido_fourier_gros(theta_arr)
        grosor       = np.clip(grosor_base + grosor_ruido, 0.10, 0.50)
        r_interior   = r_exterior * (1.0 - grosor) * ventana
        r_interior   = np.maximum(r_interior, r_cme * 0.15 * ventana)
        return r_exterior, r_interior

    def densidad_campo(self, THETA, R, r_cme, t_frame):
        """
        Devuelve el campo de densidad (NaN fuera de la CME).
        """
        r_ext_2d, r_int_2d = self.forma(THETA, r_cme)
        mascara = (R > r_int_2d) & (R <= r_ext_2d)

        dens_angular     = np.clip((1.0 + np.cos(THETA))**5, 0, None)
        r_norm           = R / (r_cme + 0.1)
        dens_radial      = np.exp(-8.0 * (r_norm - 1.0)**2)
        expansion_factor = (r_cme / self.R_CME_INICIAL)**0.5
        t_norm           = np.maximum(1.0, (t_frame - self.t_inicio_s) / 600.0)
        time_factor      = 1.0 / np.sqrt(t_norm)
        densidad_diluida = 100.0 / expansion_factor * time_factor
        dens_cme = DENSIDAD_FONDO * densidad_diluida * dens_angular * (0.3 + dens_radial)

        campo = np.where(mascara, dens_cme, np.nan)
        campo = np.where(mascara, np.clip(campo, DENSIDAD_FONDO, np.nanmax(campo)
                                          if np.any(mascara) else DENSIDAD_FONDO),
                         np.nan)
        return campo, mascara

    def campo_velocidad(self, THETA_vec, R_vec, r_cme, v_t):
        """
        Devuelve (U, V, mascara_vec) para el campo vectorial de velocidad.
        """
        r_ext_vec, r_int_vec = self.forma(THETA_vec, r_cme)
        mascara_vec = (R_vec > r_int_vec) & (R_vec <= r_ext_vec)

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
# INSTANCIAS DE CADA CME
# ──────────────────────────────────────────────────────────────────────────────
RETRASO_CME2 = 4500.0   # CME-2 sale 1.39 horas después que CME-1

cme1 = CME(
    nombre='CME-1',
    tr=138, td=1249, ar=0.001, ad=4.950,
    v0=40,  x0=25000,
    R_CME_INICIAL=2.2, 
    semilla=26,
    color_cine='steelblue',
    t_inicio_s=0.0,
)

cme2 = CME(
    nombre='CME-2',
    tr=100, td=1050, ar=0.002, ad=5.150,
    v0=50,  x0=21000,
    R_CME_INICIAL=1.1, 
    semilla=15,
    color_cine='red',
    t_inicio_s=RETRASO_CME2,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. GRÁFICAS CINEMÁTICAS CONJUNTAS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("CINEMÁTICA CONJUNTA: CME-1 y CME-2")
print("="*80)

tiempos   = np.linspace(0, T_HORAS * 3600, 500)
tiempos_h = tiempos / 3600

print("Calculando cinemática CME-1...")
_p1   = [cme1.posicion(t) for t in tiempos]
pos1  = np.array([p if p is not None else cme1.x0 for p in _p1], dtype=float)
vel1  = np.array([cme1.velocidad(t) for t in tiempos], dtype=float)
acel1 = np.array([cme1.aceleracion(max(0.0, t - cme1.t_inicio_s)) for t in tiempos], dtype=float) * 1000.0  # km/s² → m/s²

print("Calculando cinemática CME-2...")
_p2   = [cme2.posicion(t) if t >= cme2.t_inicio_s else None for t in tiempos]
pos2  = np.array([p if p is not None else np.nan for p in _p2], dtype=float)
vel2  = np.array([cme2.velocidad(t) if t >= cme2.t_inicio_s else np.nan
                  for t in tiempos], dtype=float)
acel2 = np.array([cme2.aceleracion(max(0.0, t - cme2.t_inicio_s))
                  if t >= cme2.t_inicio_s else np.nan for t in tiempos], dtype=float) * 1000.0  # km/s² → m/s²

pos1_rs = pos1 / R_SOL_KM
pos2_rs = pos2 / R_SOL_KM

# Etapas CME-1
idx_amax1 = np.argmax(acel1)
t_inic1   = tiempos_h[idx_amax1]
v_umbral1 = 0.95 * np.nanmax(vel1)
t_acel1   = tiempos_h[np.argmax(vel1 >= v_umbral1)]

# Etapas CME-2
acel2_clean = np.where(np.isnan(acel2), -np.inf, acel2)
idx_amax2   = np.argmax(acel2_clean)
t_inic2     = tiempos_h[idx_amax2]
v_umbral2   = 0.95 * np.nanmax(vel2)
valid_vel2  = np.where(~np.isnan(vel2) & (vel2 >= v_umbral2))
t_acel2     = tiempos_h[valid_vel2[0][0]] if len(valid_vel2[0]) > 0 else T_HORAS

print(f"\nCME-1 | Fin iniciación: {t_inic1:.2f} h  |  Fin aceleración: {t_acel1:.2f} h")
print(f"CME-2 | Fin iniciación: {t_inic2:.2f} h  |  Fin aceleración: {t_acel2:.2f} h")

fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                          sharex=True, gridspec_kw={'hspace': 0})
fig.suptitle('Cinemática conjunta: CME-1 y CME-2', fontsize=20, y=0.98)
fig.text(0.5, 0.935, f'{T_HORAS} horas de propagación',
         ha='center', fontsize=13, style='italic', color='#444444')

ax_pos, ax_vel, ax_acel = axes

# ── Colores de fondo en escala de grises ─────────────────────────────────────
# Iniciación y Propagación: blanco | Aceleración (ambas CMEs): gris claro
C_BLANCO = '#FFFFFF'
C_GRIS   = '#CCCCCC'

def sombrear_etapas_ambas(ax, t_i1, t_a1, t_i2, t_a2):
    """Pinta el fondo del panel: blanco base, gris en las zonas de
    aceleración de CME-1 y CME-2. Las líneas divisorias se dibujan
    para cada CME por separado."""
    # Fondo blanco base
    ax.axvspan(0, T_HORAS, color=C_BLANCO, alpha=1.0, zorder=0)
    # Zona de aceleración CME-1
    ax.axvspan(t_i1, t_a1, color=C_GRIS, alpha=0.5, zorder=0)
    # Zona de aceleración CME-2
    ax.axvspan(t_i2, t_a2, color=C_GRIS, alpha=0.5, zorder=0)
    # Líneas CME-1: discontinua ini/acel, punteada acel/prop
    ax.axvline(x=t_i1, color='k', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(x=t_a1, color='k', linestyle=':',  linewidth=0.8, alpha=0.5, zorder=2)
    # Líneas CME-2
    ax.axvline(x=t_i2, color='k', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(x=t_a2, color='k', linestyle=':',  linewidth=0.8, alpha=0.5, zorder=2)

# — Posición —
sombrear_etapas_ambas(ax_pos, t_inic1, t_acel1, t_inic2, t_acel2)
ax_pos.plot(tiempos_h, pos1_rs, color=cme1.color_cine, linewidth=2.5,
            label=cme1.nombre, zorder=3)
ax_pos.plot(tiempos_h, pos2_rs, color=cme2.color_cine, linewidth=2.5,
            label=cme2.nombre, zorder=3)
ax_pos.set_ylabel(f'Posición ({R_SOL_STR})', fontsize=16)
ax_pos.set_xlim(0, T_HORAS)
ax_pos.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax_pos.tick_params(bottom=False)
ax_pos.legend(fontsize=13, loc='upper right')

# — Velocidad —
sombrear_etapas_ambas(ax_vel, t_inic1, t_acel1, t_inic2, t_acel2)
ax_vel.plot(tiempos_h, vel1, color=cme1.color_cine, linewidth=2.5, zorder=3)
ax_vel.plot(tiempos_h, vel2, color=cme2.color_cine, linewidth=2.5, zorder=3)
ax_vel.set_ylabel('Velocidad (km/s)', fontsize=16)
ax_vel.set_xlim(0, T_HORAS)
ax_vel.grid(True, alpha=0.3, linestyle='--', zorder=1)
ax_vel.tick_params(bottom=False)

# — Aceleración —
sombrear_etapas_ambas(ax_acel, t_inic1, t_acel1, t_inic2, t_acel2)
ax_acel.plot(tiempos_h, acel1, color=cme1.color_cine, linewidth=2.5, zorder=3)
ax_acel.plot(tiempos_h, acel2, color=cme2.color_cine, linewidth=2.5, zorder=3)
ax_acel.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5, zorder=2)
ax_acel.set_xlabel('Tiempo (h)', fontsize=16)
ax_acel.set_ylabel(r'Aceleración (m/s$^2$)', fontsize=16)
ax_acel.set_xlim(0, T_HORAS)
ax_acel.set_xticks(np.arange(0, T_HORAS + 1, 1))
ax_acel.grid(True, alpha=0.3, linestyle='--', zorder=1)

# Etiquetas de fases — CME-1 arriba, CME-2 abajo para no solaparse
for ax in axes:
    ylim  = ax.get_ylim()
    rango = ylim[1] - ylim[0]
    y1 = ylim[0] + rango * 0.82
    y2 = ylim[0] + rango * 0.68
    for t_mid, label in [((0 + t_inic1)/2,       'Ini-1'),
                          ((t_inic1 + t_acel1)/2, 'Acel-1'),
                          ((t_acel1 + T_HORAS)/2, 'Prop-1')]:
        ax.text(t_mid, y1, label, ha='center', fontsize=12,
                color='#444444', zorder=4, rotation=90, va='center')
    for t_mid, label in [((cme2.t_inicio_s/3600 + t_inic2)/2, 'Ini-2'),
                          ((t_inic2 + t_acel2)/2,              'Acel-2'),
                          ((t_acel2 + T_HORAS)/2,              'Prop-2')]:
        ax.text(t_mid, y2, label, ha='center', fontsize=12,
                color='#444444', zorder=4, rotation=90, va='center')

plt.savefig("cinematica_conjunta.pdf", dpi=300, bbox_inches='tight')
print("✓ Gráfica cinemática guardada: cinematica_conjunta.pdf")
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 2. PROPAGACIÓN POLAR CONJUNTA (8 FRAMES)
#    CME-1 lanza en t=0, CME-2 lanza en t=1 h
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("VISUALIZACIÓN POLAR CONJUNTA: CME-1 y CME-2")
print("="*80)

ESTADOS_TIEMPO = 8
ETIQUETAS      = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# Frames distribuidos de 1 h a 10 h (tiempo global)
tiempos_frames = np.linspace(3600, T_HORAS * 3600, ESTADOS_TIEMPO)

# Radio máximo global para escalar los últimos frames
pos_final_c1    = cme1.posicion(tiempos_frames[-1]) / FACTOR_ESCALA
pos_final_c2    = cme2.posicion(tiempos_frames[-1]) / FACTOR_ESCALA
LIMITE_RADIO_MAX = max(pos_final_c1, pos_final_c2) * 1.4

# Malla polar global — alta resolución para bordes suaves
theta    = np.linspace(-np.pi, np.pi, 800)
r        = np.linspace(0, LIMITE_RADIO_MAX, 400)
THETA, R = np.meshgrid(theta, r)

# Malla para vectores de velocidad — mayor densidad
theta_vec          = np.linspace(-np.pi, np.pi, 30)
r_vec              = np.linspace(0.5, LIMITE_RADIO_MAX, 8)
THETA_vec, R_vec   = np.meshgrid(theta_vec, r_vec)

# ── Pre-cálculo del máximo global de densidad (incluyendo solapamiento) ──────
print("  Pre-calculando máximo global de densidad...")
dens_max_global = DENSIDAD_FONDO  # valor mínimo de referencia

for t_pre in tiempos_frames:
    r_pre1 = cme1.posicion(t_pre) / FACTOR_ESCALA
    c_pre1, m_pre1 = cme1.densidad_campo(THETA, R, r_pre1, t_pre)

    activa_pre2 = t_pre >= cme2.t_inicio_s
    if activa_pre2:
        r_pre2 = cme2.posicion(t_pre) / FACTOR_ESCALA
        c_pre2, m_pre2 = cme2.densidad_campo(THETA, R, r_pre2, t_pre)
    else:
        c_pre2 = np.full(R.shape, np.nan)
        m_pre2 = np.zeros(R.shape, dtype=bool)

    solapado_pre = m_pre1 & m_pre2
    solo1_pre    = m_pre1 & ~m_pre2
    solo2_pre    = ~m_pre1 & m_pre2

    d_pre = np.full(R.shape, np.nan)
    d_pre = np.where(solo1_pre,    c_pre1,            d_pre)
    d_pre = np.where(solo2_pre,    c_pre2,            d_pre)
    d_pre = np.where(solapado_pre, c_pre1 + c_pre2,   d_pre)

    frame_max = float(np.nanmax(d_pre)) if np.any(~np.isnan(d_pre)) else DENSIDAD_FONDO
    if frame_max > dens_max_global:
        dens_max_global = frame_max

DENS_MIN_GLOBAL = float(np.log10(DENSIDAD_FONDO))
DENS_MAX_GLOBAL = max(float(np.log10(dens_max_global)), DENS_MIN_GLOBAL + 0.1)
print(f"  Rango de densidad global: [{DENS_MIN_GLOBAL:.3f}, {DENS_MAX_GLOBAL:.3f}] log10")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('Propagación conjunta: CME-1 y CME-2',
             fontsize=18, fontweight='normal', y=0.99)
fig.text(0.5, 0.935, f'{T_HORAS} horas de propagación',
         ha='center', fontsize=13, style='italic', color='#444444')

for idx, t_frame in enumerate(tiempos_frames):
    print(f"  Frame {idx+1}/{ESTADOS_TIEMPO}: t = {t_frame/3600:.1f} h", end=" ... ")

    ax = plt.subplot(2, 4, idx + 1, projection='polar')
    ax.text(0.05, 0.97, f'{ETIQUETAS[idx]})',
            transform=ax.transAxes, fontsize=12, fontweight='normal',
            va='top', ha='left', color='black', fontstyle='italic')

    # ── Malla local adaptativa para frames pequeños (primeros 4) ───────────────
    r_cme1 = cme1.posicion(t_frame) / FACTOR_ESCALA
    v_cme1 = cme1.velocidad(t_frame)
    activa_cme2 = t_frame >= cme2.t_inicio_s
    if activa_cme2:
        r_cme2 = cme2.posicion(t_frame) / FACTOR_ESCALA
        v_cme2 = cme2.velocidad(t_frame)
    else:
        r_cme2 = None
        v_cme2 = 0.0

    if idx < 4:
        r_max_local = max(r_cme1, r_cme2 if r_cme2 else 0) * 2.7
        r_max_local = max(r_max_local, 1.0)
        th_loc = np.linspace(-np.pi, np.pi, 800)
        r_loc  = np.linspace(0, r_max_local, 400)
        TH_loc, R_loc = np.meshgrid(th_loc, r_loc)
        tv_loc = np.linspace(-np.pi, np.pi, 30)
        rv_loc = np.linspace(0.1, r_max_local, 8)
        THv_loc, Rv_loc = np.meshgrid(tv_loc, rv_loc)
    else:
        TH_loc, R_loc   = THETA, R
        THv_loc, Rv_loc = THETA_vec, R_vec

    # ── Fondo azul claro ─────────────────────────────────────────────────────
    ax.contourf(TH_loc, R_loc,
                np.ones_like(R_loc),
                levels=[0.5, 1.5], colors=['#D6EAF8'], alpha=0.8)

    # ── Campos de densidad por CME ────────────────────────────────────────────
    dens_total = np.full(R_loc.shape, np.nan)

    campo1, mask1 = cme1.densidad_campo(TH_loc, R_loc, r_cme1, t_frame)

    # CME-2 (existe solo si t_frame >= t_inicio)
    if activa_cme2:
        campo2, mask2 = cme2.densidad_campo(TH_loc, R_loc, r_cme2, t_frame)
    else:
        campo2 = np.full(R_loc.shape, np.nan)
        mask2  = np.zeros(R_loc.shape, dtype=bool)

    # Suma de densidad donde se solapan
    solo1    = mask1 & ~mask2
    solo2    = ~mask1 & mask2
    solapado = mask1 & mask2

    dens_total = np.where(solo1,    campo1,           dens_total)
    dens_total = np.where(solo2,    campo2,           dens_total)
    dens_total = np.where(solapado, campo1 + campo2,  dens_total)

    # Escala logarítmica con rango global fijo
    dens_log = np.log10(np.where(~np.isnan(dens_total),
                                  np.maximum(dens_total, DENSIDAD_FONDO / 10.0),
                                  np.nan))

    levels_dens = np.linspace(DENS_MIN_GLOBAL, DENS_MAX_GLOBAL, 100)
    ax.contourf(TH_loc, R_loc, dens_log, levels=levels_dens, cmap='viridis', alpha=0.9)

    # ── Vectores de velocidad ─────────────────────────────────────────────────
    # CME-1
    U1, V1, mv1 = cme1.campo_velocidad(THv_loc, Rv_loc, r_cme1, v_cme1)

    # CME-2
    if activa_cme2:
        U2, V2, mv2 = cme2.campo_velocidad(THv_loc, Rv_loc, r_cme2, v_cme2)
    else:
        U2  = np.full(THv_loc.shape, np.nan)
        V2  = np.full(THv_loc.shape, np.nan)
        mv2 = np.zeros(THv_loc.shape, dtype=bool)

    solo_v1    = mv1 & ~mv2
    solo_v2    = ~mv1 & mv2
    solapado_v = mv1 & mv2

    # En zonas solapadas: suma vectorial de velocidades (normalizada al plotear)
    U_plot = np.where(solo_v1,    U1,
             np.where(solo_v2,    U2,
             np.where(solapado_v, U1 + U2, np.nan)))
    V_plot = np.where(solo_v1,    V1,
             np.where(solo_v2,    V2,
             np.where(solapado_v, V1 + V2, np.nan)))

    ax.quiver(THv_loc, Rv_loc, U_plot, V_plot,
              scale=20, width=0.004, color='red', alpha=0.9)

    # ── Título del frame ──────────────────────────────────────────────────────
    if activa_cme2:
        titulo = (f"t = {t_frame/3600:.1f} h  |  "
                  f"r₁={r_cme1:.1f} {R_SOL_STR}  r₂={r_cme2:.1f} {R_SOL_STR}")
    else:
        titulo = f"t = {t_frame/3600:.1f} h  |  r₁={r_cme1:.1f} {R_SOL_STR}  (CME-2 aún no)"
    ax.set_title(titulo, fontsize=8, fontweight='normal', pad=10)

    # ── Escala radial ─────────────────────────────────────────────────────────
    if idx < 4:
        limite_frame = max(r_cme1,
                           r_cme2 if activa_cme2 and r_cme2 else 0) * 2.7
        ax.set_ylim([0, max(limite_frame, 1.0)])
    else:
        ax.set_ylim([0, LIMITE_RADIO_MAX])

    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

    status = f"r₁={r_cme1:.2f}"
    if activa_cme2:
        status += f"  r₂={r_cme2:.2f} R_sol"
    print(status + " ✓")

plt.tight_layout(rect=[0, 0, 0.92, 0.97])

# Barra de color
cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.75])
sm = plt.cm.ScalarMappable(cmap='viridis',
                            norm=plt.Normalize(vmin=DENS_MIN_GLOBAL, vmax=DENS_MAX_GLOBAL))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'log$_{10}$($\rho$) [protones/cm$^3$]',
               rotation=270, labelpad=25, fontsize=11)

plt.savefig("cme_conjunta_polar.pdf", dpi=300, bbox_inches='tight')
print("\n✓ Visualización polar guardada: cme_conjunta_polar.pdf")
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 3. RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)

for cme, vels, acels, poss in [
    (cme1, vel1, acel1, pos1),
    (cme2, vel2, acel2, pos2),
]:
    print(f"\n  {cme.nombre}  (lanzamiento en t = {cme.t_inicio_s/3600:.1f} h):")
    print(f"    Velocidad inicial:    {cme.v0:.2f} km/s")
    print(f"    Velocidad máxima:     {np.nanmax(vels):.2f} km/s")
    print(f"    Velocidad final:      {vels[~np.isnan(vels)][-1]:.2f} km/s")
    print(f"    Aceleración máxima:   {np.nanmax(acels):.4f} m/s²")
    print(f"    Posición inicial:     {cme.x0 / R_SOL_KM:.4f} R_sol")
    pos_fin = poss[~np.isnan(poss)][-1]
    print(f"    Posición final:       {pos_fin / R_SOL_KM:.4f} R_sol")
    print(f"    Distancia recorrida:  {(pos_fin - cme.x0) / R_SOL_KM:.4f} R_sol")

print("\n" + "="*80)
print("✓ SIMULACIÓN CONJUNTA COMPLETADA")
print("="*80)
