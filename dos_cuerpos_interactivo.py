"""
Simulador de dos cuerpos — a(t), v(t), x(t)
=============================================
a(t) = [ 1/(a_r * exp(t/tau_r))  +  1/(a_d * exp(-t/tau_d)) ]^(-1)

Caracteristicas:
  - RK4 vectorizado, N=1500
  - T_max dinamico: si ambas curvas son visibles y se cruzan, se ajusta
    automaticamente con margen y se re-integra (sin ciclos de desfase)
  - a_r/a_d se ingresan en km/s², grafica de aceleracion en m/s²
  - Posicion con tres ejes: km, AU y radios solares (R_sol)
  - Interseccion de x(t): marca la primera, con t [h], x [km] y [AU]
  - Carga/Exportacion de parametros via CSV
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button, RadioButtons
import csv, os
from datetime import datetime
from tkinter import filedialog, Tk

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
S2H     = 1.0 / 3600.0
KM2AU   = 1.0 / 1.496e8       # 1 AU      = 1.496e8 km
KM2RSOL = 1.0 / 696000.0      # 1 R_sol   = 696000 km
KM2M_S2 = 1000.0              # km/s^2 -> m/s^2
N       = 1500

BG    = '#1c1c1e'
SURF  = '#2c2c2e'
GRID  = '#3a3a3c'
TEXT  = '#ebebf0'
MUTED = '#888'

COLORS = [
    ('#5b9cf6', '#ff7b54', '#57cc99'),
    ('#ffd60a', '#c084fc', '#fb923c'),
]

DEFAULTS = [
    dict(ar=1.0, tr=2.0, ad=1.0, td=2.0, v0=0.0, x0=0.0, label="Cuerpo 1"),
    dict(ar=2.0, tr=4.0, ad=0.5, td=3.0, v0=0.0, x0=0.0, label="Cuerpo 2"),
]
T_MAX_DEFAULT    = 30.0
T_OFFSET_DEFAULT = 0.0

FIELDS = ['ar',    'tr',    'ad',    'td',    'v0',   'x0']
LABELS = ['a_r',   'tau_r', 'a_d',   'tau_d', 'v_0',  'x_0']
UNITS  = ['km/s²', 's',     'km/s²', 's',     'km/s', 'km']


# ─────────────────────────────────────────────────────────────────────────────
# Fisica
# ─────────────────────────────────────────────────────────────────────────────
def _a(t, ar, tr, ad, td):
    er = ar * np.exp(np.clip( t / tr, -300, 300))
    ed = ad * np.exp(np.clip(-t / td, -300, 300))
    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.where((er > 0) & (ed > 0), 1.0 / (1.0/er + 1.0/ed), 0.0)
    return np.where(np.isfinite(val), val, 0.0)


def integrar(ar, tr, ad, td, v0, x0, t0, T_max):
    """
    RK4 vectorizado. Integra desde t=t0 hasta t=T_max (tiempo absoluto,
    en segundos). Devuelve arrays de tiempo absoluto y a/v/x.
    """
    if t0 >= T_max:
        a0 = _a(np.array([0.0]), ar, tr, ad, td)
        return (np.array([t0]), a0, np.array([v0]), np.array([x0]))

    tl = np.linspace(0.0, T_max - t0, N)   # tiempo local (relativo a t0)
    dt = tl[1] - tl[0]

    k1 = _a(tl,          ar, tr, ad, td)
    k2 = _a(tl + dt/2,   ar, tr, ad, td)
    k4 = _a(tl + dt,     ar, tr, ad, td)
    # k3 == k2 porque a(t) no depende del estado, solo del tiempo
    dv = (k1 + 4.0*k2 + k4) * (dt / 6.0)

    v = np.empty(N); v[0] = v0
    v[1:] = v0 + np.cumsum(dv[:-1])

    v_mid = 0.5 * (v[:-1] + v[1:])
    x = np.empty(N); x[0] = x0
    x[1:] = x0 + np.cumsum(dt * v_mid)

    a_final = _a(tl, ar, tr, ad, td)
    return tl + t0, a_final, v, x


def primera_interseccion(t1, x1, t2, x2):
    """Interpola linealmente la primera vez que x1(t) == x2(t).
    Devuelve (t_h, x_km) o (None, None) si no hay cruce."""
    t_min = max(t1[0],  t2[0])
    t_max = min(t1[-1], t2[-1])
    if t_min >= t_max:
        return None, None
    tc   = np.linspace(t_min, t_max, max(len(t1), len(t2)))
    xc1  = np.interp(tc, t1, x1)
    xc2  = np.interp(tc, t2, x2)
    diff = xc1 - xc2
    signs = np.sign(diff)
    cross = np.where(np.diff(signs) != 0)[0]
    if len(cross) == 0:
        return None, None
    i = cross[0]
    d0, d1 = diff[i], diff[i+1]
    frac   = -d0 / (d1 - d0)
    t_int  = tc[i] + frac * (tc[i+1] - tc[i])
    x_int  = xc1[i] + frac * (xc1[i+1] - xc1[i])
    return t_int * S2H, x_int   # (horas, km)


# ─────────────────────────────────────────────────────────────────────────────
# Estado
# ─────────────────────────────────────────────────────────────────────────────
params  = [d.copy() for d in DEFAULTS]
visible = [True, True]
data    = [None, None]
data_cache = [None, None]   # (t_h, a_m_s2, v, x) por cuerpo
state   = dict(
    T_max          = T_MAX_DEFAULT,   # T_max EFECTIVO usado para integrar/graficar
    t_max_usuario  = T_MAX_DEFAULT,   # T_max introducido manualmente por el usuario
    offset_s       = T_OFFSET_DEFAULT,
    offset_on      = 1,
    auto_tmax      = True,            # si True, T_max se extiende automaticamente
                                       # hasta la interseccion cuando ambas son visibles
)


def t_start(idx):
    return state['offset_s'] if idx == state['offset_on'] else 0.0


def recalc(idx):
    p = params[idx]
    try:
        data[idx] = integrar(p['ar'], p['tr'], p['ad'], p['td'],
                             p['v0'], p['x0'], t_start(idx), state['T_max'])
        data_cache[idx] = None
    except Exception as e:
        print(f"[recalc C{idx+1}] {e}")


def get_cached_data(idx):
    """Datos con conversion de unidades precomputada (cache invalidado por recalc)."""
    if data_cache[idx] is not None:
        return data_cache[idx]
    if data[idx] is None:
        return None
    t_s, a, v, x = data[idx]
    t_h    = t_s * S2H
    a_ms2  = a * KM2M_S2
    data_cache[idx] = (t_h, a_ms2, v, x)
    return data_cache[idx]


def calc_t_max_dynamic():
    """
    Calcula el T_max EFECTIVO:
      - Si auto_tmax esta activo, y las dos curvas son visibles y se
        intersectan, usa max(t_interseccion * 1.1, t_max_usuario)
      - En cualquier otro caso, usa t_max_usuario.
    Requiere que data[] ya este poblado con el T_max anterior
    (se usa solo como estimacion para decidir si hay que extender).
    """
    n_visible = sum(visible)
    base = state['t_max_usuario']

    if state['auto_tmax'] and n_visible == 2 and data[0] is not None and data[1] is not None:
        t_i_h, _ = primera_interseccion(data[0][0], data[0][3],
                                         data[1][0], data[1][3])
        if t_i_h is not None:
            t_i_s = t_i_h / S2H
            return max(t_i_s * 1.1, base)

    return base


def recalc_all_with_dynamic_tmax():
    """
    Re-integra ambos cuerpos con el T_max actual, calcula el T_max
    dinamico, y si cambia, vuelve a re-integrar con el nuevo valor.
    Evita el desfase de un paso entre datos y T_max mostrado.
    """
    recalc(0); recalc(1)
    new_tmax = calc_t_max_dynamic()
    if abs(new_tmax - state['T_max']) > 1e-9:
        state['T_max'] = new_tmax
        recalc(0); recalc(1)


recalc_all_with_dynamic_tmax()


# ─────────────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor(BG)
try:
    fig.canvas.manager.set_window_title("Simulador dos cuerpos")
except Exception:
    pass

outer = gridspec.GridSpec(2, 1, figure=fig,
                          top=0.95, bottom=0.02, left=0.06, right=0.88,
                          hspace=0.06, height_ratios=[3.4, 1.0])
gs_plots = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0], hspace=0.42)
ax_a = fig.add_subplot(gs_plots[0])
ax_v = fig.add_subplot(gs_plots[1])
ax_x = fig.add_subplot(gs_plots[2])

# Eje derecho AU para posicion
ax_x_au = ax_x.twinx()
ax_x_au.spines['right'].set_position(('outward', 0))
ax_x_au.set_facecolor('none')
ax_x_au.tick_params(colors=MUTED, labelsize=8)
ax_x_au.set_ylabel('x(t)  [AU]', color=MUTED, fontsize=9)
for sp in ax_x_au.spines.values():
    sp.set_edgecolor(GRID)

# Eje extremo derecho en radios solares (espacio reservado con right=0.88)
ax_x_rsol = ax_x.twinx()
ax_x_rsol.spines['right'].set_position(('outward', 62))
ax_x_rsol.set_facecolor('none')
ax_x_rsol.tick_params(colors=MUTED, labelsize=8)
ax_x_rsol.set_ylabel('x(t)  [R☉]', color=MUTED, fontsize=9)
for sp in ax_x_rsol.spines.values():
    sp.set_edgecolor(GRID)

# Reposicionar los exponentes "1eN" de cada eje para que no se solapen:
#   ax_x  (km, izquierda)  -> arriba a la izquierda (posicion por defecto, sin cambio)
#   ax_x_au (AU)           -> desplazado hacia arriba
#   ax_x_rsol (R_sol)      -> desplazado mas arriba aun, hacia la derecha
ax_x_au.yaxis.get_offset_text().set_position((1.04, 1.10))
ax_x_au.yaxis.get_offset_text().set_color(MUTED)
ax_x_au.yaxis.get_offset_text().set_fontsize(7.5)

ax_x_rsol.yaxis.get_offset_text().set_position((1.10, 1.22))
ax_x_rsol.yaxis.get_offset_text().set_color(MUTED)
ax_x_rsol.yaxis.get_offset_text().set_fontsize(7.5)

ax_x.yaxis.get_offset_text().set_color(MUTED)
ax_x.yaxis.get_offset_text().set_fontsize(7.5)


gs_ctrl = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1],
                                           wspace=0.05, width_ratios=[1, 1, 0.72])
for ax in [fig.add_subplot(gs_ctrl[i]) for i in range(3)]:
    ax.set_facecolor(SURF); ax.axis('off')
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

for ax in [ax_a, ax_v, ax_x]:
    ax.set_facecolor(SURF)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)


# ─────────────────────────────────────────────────────────────────────────────
# Lineas
# ─────────────────────────────────────────────────────────────────────────────
lines  = {}
dot_vm = []

for idx in range(2):
    ca, cv, cx = COLORS[idx]
    ls  = '-'   if idx == 0 else '--'
    lw  = 2.0   if idx == 0 else 1.8
    lbl = params[idx]['label']
    lines[f'a{idx}'], = ax_a.plot([], [], color=ca, lw=lw, ls=ls, label=lbl)
    lines[f'v{idx}'], = ax_v.plot([], [], color=cv, lw=lw, ls=ls, label=lbl)
    lines[f'x{idx}'], = ax_x.plot([], [], color=cx, lw=lw, ls=ls, label=lbl)
    dot, = ax_v.plot([], [], 'o', color=ca, ms=7, zorder=6, label='_')
    dot_vm.append(dot)

# Marcador de interseccion
dot_inter, = ax_x.plot([], [], '*', color='#ffffff', ms=12, zorder=7,
                        markeredgecolor='#aaa', markeredgewidth=0.5, label='_')
txt_inter  = ax_x.text(0.5, 0.96, '', transform=ax_x.transAxes,
                        color='#ffffff', fontsize=7.5, va='top', ha='center',
                        bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec='none', alpha=0.85))

def _estilo(ax, ylabel, title):
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel('tiempo  [h]', color=MUTED, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=9, pad=4)
    ax.grid(color=GRID, lw=0.5, ls='--')
    ax.legend(facecolor='none', edgecolor='none', labelcolor=TEXT,
              fontsize=8, loc='upper left', framealpha=0.0,
              borderpad=0.4, handlelength=1.6)

_estilo(ax_a, 'a(t)  [m/s²]', 'Aceleracion')
_estilo(ax_v, 'v(t)  [km/s]', 'Velocidad')
_estilo(ax_x, 'x(t)  [km]',   'Posicion')

# Textos de a(0) y v(0) — esquina superior derecha de cada grafica
txt_a0 = [
    ax_a.text(0.98, 0.97 - 0.10*i, '', transform=ax_a.transAxes,
              color=COLORS[i][0], fontsize=7.5, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec='none', alpha=0.85))
    for i in range(2)
]
txt_v0 = [
    ax_v.text(0.98, 0.97 - 0.10*i, '', transform=ax_v.transAxes,
              color=COLORS[i][0], fontsize=7.5, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec='none', alpha=0.85))
    for i in range(2)
]
# 95% v_max — esquina inferior derecha de velocidad
txt_vmax = [
    ax_v.text(0.99, 0.03 + 0.11*i, '', transform=ax_v.transAxes,
              color=COLORS[i][0], fontsize=7.5, va='bottom', ha='right',
              bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec='none', alpha=0.85))
    for i in range(2)
]

fig.suptitle(
    'a(t) = [ 1/(a_r · exp(t/tau_r))  +  1/(a_d · exp(-t/tau_d)) ]^(-1)'
    '        |        dos cuerpos  —  ejes en horas',
    color=TEXT, fontsize=10.5, fontweight='500'
)


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────
def render():
    Th = state['T_max'] * S2H

    for idx in range(2):
        vis = visible[idx]
        for key in [f'a{idx}', f'v{idx}', f'x{idx}']:
            lines[key].set_visible(vis)
        dot_vm[idx].set_visible(vis)
        txt_a0[idx].set_visible(vis)
        txt_v0[idx].set_visible(vis)
        txt_vmax[idx].set_visible(vis)

        if vis:
            cached = get_cached_data(idx)
            if cached is not None:
                t_h, a_ms2, v, x = cached
                lbl = params[idx]['label']
                off_h = t_start(idx) * S2H
                if off_h > 0:
                    lbl += f'  (offset={off_h:.4f} h)'

                lines[f'a{idx}'].set_data(t_h, a_ms2)
                lines[f'v{idx}'].set_data(t_h, v)
                lines[f'x{idx}'].set_data(t_h, x)
                lines[f'a{idx}'].set_label(lbl)
                lines[f'v{idx}'].set_label(lbl)
                lines[f'x{idx}'].set_label(lbl)

                txt_a0[idx].set_text(f'C{idx+1}  a(0) = {a_ms2[0]:.4f} m/s²')
                txt_v0[idx].set_text(f'C{idx+1}  v(0) = {v[0]:.4f} km/s')

                vmax = v.max()
                i95  = int(np.argmax(v >= 0.95 * vmax))
                dot_vm[idx].set_data([t_h[i95]], [v[i95]])
                txt_vmax[idx].set_text(
                    f'C{idx+1}  95% v_max = {v[i95]:.4f} km/s   @  t = {t_h[i95]:.5f} h')

    # ── Interseccion de posicion (solo si ambas curvas son visibles) ─────────
    show_inter = (visible[0] and visible[1] and
                  data[0] is not None and data[1] is not None)
    if show_inter:
        t_i, x_i = primera_interseccion(data[0][0], data[0][3],
                                         data[1][0], data[1][3])
        if t_i is not None:
            dot_inter.set_data([t_i], [x_i])
            dot_inter.set_visible(True)
            txt_inter.set_text(
                f'Interseccion:  t = {t_i:.5f} h'
                f'   x = {x_i:.4f} km  =  {x_i*KM2AU:.4e} AU')
            txt_inter.set_visible(True)
        else:
            dot_inter.set_data([], [])
            dot_inter.set_visible(False)
            txt_inter.set_text('')
    else:
        dot_inter.set_data([], [])
        dot_inter.set_visible(False)
        txt_inter.set_text('')

    # ── Escalado de ejes (tiempo) ─────────────────────────────────────────────
    for ax in [ax_a, ax_v, ax_x]:
        ax.set_xlim(0, Th)
        ax.relim()
        ax.autoscale_view(scalex=False)

    # Aceleracion: escala segun maximo visible, con margen
    a_values = []
    for idx in range(2):
        if visible[idx]:
            cached = get_cached_data(idx)
            if cached is not None:
                _, a_ms2, _, _ = cached
                a_values.append(a_ms2.max())
    if a_values:
        ax_a.set_ylim(0, max(a_values) * 1.1)

    # Sincronizar eje AU y R_sol con el eje km de posicion
    ymin, ymax = ax_x.get_ylim()
    ax_x_au.set_ylim(ymin * KM2AU, ymax * KM2AU)
    ax_x_au.set_xlim(0, Th)
    ax_x_rsol.set_ylim(ymin * KM2RSOL, ymax * KM2RSOL)
    ax_x_rsol.set_xlim(0, Th)

    fig.canvas.draw_idle()

render()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers widgets
# ─────────────────────────────────────────────────────────────────────────────
_refs = []

def add_textbox(rect, initial):
    global _refs
    ax_tb = fig.add_axes(rect)
    ax_tb.set_facecolor(GRID)
    for sp in ax_tb.spines.values(): sp.set_edgecolor('#555')
    tb = TextBox(ax_tb, '', initial=str(initial),
                 color=GRID, hovercolor='#4a4a4c', label_pad=0.0)
    tb.text_disp.set_color(TEXT); tb.text_disp.set_fontsize(8.5)
    _refs += [ax_tb, tb]
    return tb

def add_button(rect, label, bg='#3a3a3c', hov='#555'):
    global _refs
    ax_b = fig.add_axes(rect)
    btn  = Button(ax_b, label, color=bg, hovercolor=hov)
    btn.label.set_color(TEXT); btn.label.set_fontsize(8)
    _refs += [ax_b, btn]
    return btn


# ─────────────────────────────────────────────────────────────────────────────
# Panel parametros
# ─────────────────────────────────────────────────────────────────────────────
Y_TOP = 0.19; H_ROW = 0.030; GAP = 0.004
W_LBL = 0.038; W_BOX = 0.07
PAD_L = [0.04, 0.225]
textboxes = [[], []]

for idx in range(2):
    xL = PAD_L[idx]
    fig.text(xL, Y_TOP + 0.012, params[idx]['label'],
             color=COLORS[idx][0], fontsize=10, fontweight='500')

    for row, (field, label, unit) in enumerate(zip(FIELDS, LABELS, UNITS)):
        y_row = Y_TOP - row * (H_ROW + GAP)
        fig.text(xL, y_row + 0.009, label, color=MUTED, fontsize=8.5, va='center')
        tb = add_textbox([xL + W_LBL + 0.003, y_row, W_BOX, H_ROW - 0.003],
                         params[idx][field])
        textboxes[idx].append(tb)
        fig.text(xL + W_LBL + 0.003 + W_BOX + 0.005,
                 y_row + 0.009, unit, color='#aaa', fontsize=7.5, va='center')

        def make_cb(i, f, tb_ref):
            def cb(text):
                try:
                    val = float(text.strip())
                    if f in ('tr', 'td') and val <= 0: raise ValueError
                    if params[i][f] != val:
                        params[i][f] = val
                        recalc_all_with_dynamic_tmax()
                        tb_tmax.set_val(str(state['T_max']))
                        lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")
                        render()
                except ValueError:
                    tb_ref.set_val(str(params[i][f]))
            return cb

        tb.on_submit(make_cb(idx, field, tb))


# ─────────────────────────────────────────────────────────────────────────────
# Panel extra
# ─────────────────────────────────────────────────────────────────────────────
XE = 0.395; YE = Y_TOP - 0.014
XB = XE + 0.3   # columna derecha del panel extra

# ── T_max ─────────────────────────────────────────────────────────────────────
fig.text(XE, YE + 0.010, 'T_max', color=MUTED, fontsize=8.5, va='center')
fig.text(XE + 0.044, YE + 0.010, '[s]', color='#aaa', fontsize=7.5, va='center')
tb_tmax = add_textbox([XE + 0.065, YE, 0.082, 0.027], state['T_max'])
lbl_tmax_h = fig.text(XE + 0.150, YE + 0.009,
                      f'= {state["T_max"]*S2H:.5f} h',
                      color=COLORS[0][0], fontsize=7.5, va='center')

def cb_tmax(text):
    try:
        val = float(text.strip())
        if val <= 0: raise ValueError
        if state['t_max_usuario'] != val:
            state['t_max_usuario'] = val
            state['T_max'] = val
            recalc_all_with_dynamic_tmax()
            tb_tmax.set_val(str(state['T_max']))
            lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")
            render()
    except ValueError:
        tb_tmax.set_val(str(state['T_max']))

tb_tmax.on_submit(cb_tmax)

# ── t_offset ──────────────────────────────────────────────────────────────────
YE2 = YE - (H_ROW + GAP)
fig.text(XE, YE2 + 0.010, 't_offset', color=MUTED, fontsize=8.5, va='center')
fig.text(XE + 0.057, YE2 + 0.010, '[s]', color='#aaa', fontsize=7.5, va='center')
tb_off = add_textbox([XE + 0.065, YE2, 0.082, 0.027], state['offset_s'])
lbl_off_h = fig.text(XE + 0.150, YE2 + 0.009,
                     f'= {state["offset_s"]*S2H:.5f} h',
                     color=COLORS[1][0], fontsize=7.5, va='center')

def cb_off(text):
    try:
        val = float(text.strip())
        if val < 0: raise ValueError
        if state['offset_s'] != val:
            state['offset_s'] = val
            lbl_off_h.set_text(f'= {val*S2H:.5f} h')
            recalc_all_with_dynamic_tmax()
            tb_tmax.set_val(str(state['T_max']))
            lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")
            render()
    except ValueError:
        tb_off.set_val(str(state['offset_s']))

tb_off.on_submit(cb_off)

# ── Selector offset — columna derecha, fila superior ──────────────────────────
fig.text(XB + 0.010, YE + 0.012, 'Aplicar offset a:', color=MUTED, fontsize=8, va='center')
ax_radio = fig.add_axes([XB, YE - 0.052, 0.130, 0.055])
ax_radio.set_facecolor(SURF)
for sp in ax_radio.spines.values(): sp.set_edgecolor(GRID)
radio = RadioButtons(ax_radio, ('Cuerpo 1', 'Cuerpo 2'),
                     active=state['offset_on'], activecolor=COLORS[1][0])
for lbl in radio.labels:
    lbl.set_color(TEXT); lbl.set_fontsize(8.5)
for circ in radio.circles:
    circ.set_radius(0.12)
_refs += [ax_radio, radio]

def cb_radio(label):
    prev = state['offset_on']
    new_on = 0 if label == 'Cuerpo 1' else 1
    if prev != new_on:
        state['offset_on'] = new_on
        recalc_all_with_dynamic_tmax()
        tb_tmax.set_val(str(state['T_max']))
        lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")
        render()

radio.on_clicked(cb_radio)

# ── Botones visibilidad — columna derecha, debajo del radio ───────────────────
YE4 = YE - 0.095
btn_vis = []
for idx in range(2):
    b = add_button([XB + idx*0.118, YE4, 0.106, 0.027],
                   f'Ocultar C{idx+1}', '#3a3a3c', '#555')
    btn_vis.append(b)

    def make_toggle(i, br):
        def toggle(ev):
            visible[i] = not visible[i]
            br.label.set_text(f'{"Mostrar" if not visible[i] else "Ocultar"} C{i+1}')
            recalc_all_with_dynamic_tmax()
            tb_tmax.set_val(str(state['T_max']))
            lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")
            render()
        return toggle

    b.on_clicked(make_toggle(idx, b))

# ── Auto-ajuste T_max — fila propia, ancho total ──────────────────────────────
YE_AUTO = YE4 - 0.034
_auto_label = lambda: f"Auto-ajuste T_max: {'ON' if state['auto_tmax'] else 'OFF'}"
btn_auto = add_button([XE, YE_AUTO, 0.165, 0.027],
                      _auto_label(),
                      '#1a4a2a' if state['auto_tmax'] else '#3a3a3c',
                      '#2a6a3a' if state['auto_tmax'] else '#555')

def toggle_auto(ev):
    state['auto_tmax'] = not state['auto_tmax']
    btn_auto.label.set_text(_auto_label())
    if state['auto_tmax']:
        btn_auto.ax.set_facecolor('#1a4a2a')
        btn_auto.color = '#1a4a2a'; btn_auto.hovercolor = '#2a6a3a'
    else:
        btn_auto.ax.set_facecolor('#3a3a3c')
        btn_auto.color = '#3a3a3c'; btn_auto.hovercolor = '#555'
    recalc_all_with_dynamic_tmax()
    tb_tmax.set_val(str(state['T_max']))
    lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")
    render()

btn_auto.on_clicked(toggle_auto)

# ── Exportar / Cargar / Reset — fila inferior, ancho total ─────────────────────
YE5 = YE_AUTO - 0.034
W3  = 0.080
btn_exp  = add_button([XE,                YE5, W3,    0.027], 'Exportar CSV', '#1a3a5c', '#2255aa')
btn_load = add_button([XE + W3 + 0.005,   YE5, W3,    0.027], 'Cargar CSV',   '#1a3a1a', '#225522')
btn_rst  = add_button([XE + 2*(W3+0.005), YE5, 0.065, 0.027], 'Reset',        '#3a2010', '#7a4010')

ax_msg  = fig.add_axes([XE, YE5 - 0.028, 0.28, 0.022])
ax_msg.axis('off')
txt_msg = ax_msg.text(0, 0.5, '', color='#aaa', fontsize=7.5, va='center')
_refs.append(ax_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Exportar CSV — guarda solo parametros (reproducible)
# ─────────────────────────────────────────────────────────────────────────────
def exportar(ev):
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'dos_cuerpos_{ts}.csv'
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parametro', 'valor'])
        w.writerow(['T_max_usuario_s', state['t_max_usuario']])
        w.writerow(['offset_s',        state['offset_s']])
        w.writerow(['offset_on',       state['offset_on']])
        for idx in range(2):
            p = params[idx]
            prefix = f'c{idx+1}'
            for field in FIELDS:
                w.writerow([f'{prefix}_{field}', p[field]])
    txt_msg.set_text(f'Guardado: {os.path.basename(fname)}')
    fig.canvas.draw_idle()

btn_exp.on_clicked(exportar)


# ─────────────────────────────────────────────────────────────────────────────
# Cargar CSV
# ─────────────────────────────────────────────────────────────────────────────
def cargar(ev):
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    fname = filedialog.askopenfilename(
        title='Abrir CSV de dos cuerpos',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
    )
    root.destroy()
    if not fname:
        return

    try:
        cfg = {}
        with open(fname, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cfg[row['parametro']] = float(row['valor'])

        t_max_usr = cfg.get('T_max_usuario_s', cfg.get('T_max_s', state['t_max_usuario']))
        state['t_max_usuario'] = t_max_usr
        state['T_max']         = t_max_usr
        state['offset_s']      = cfg.get('offset_s', state['offset_s'])
        state['offset_on']     = int(cfg.get('offset_on', state['offset_on']))

        tb_off.set_val(str(state['offset_s']))
        lbl_off_h.set_text(f"= {state['offset_s']*S2H:.5f} h")
        radio.set_active(state['offset_on'])

        for idx in range(2):
            prefix = f'c{idx+1}'
            for field in FIELDS:
                key = f'{prefix}_{field}'
                if key in cfg:
                    params[idx][field] = cfg[key]
            for row, field in enumerate(FIELDS):
                textboxes[idx][row].set_val(str(params[idx][field]))

        recalc_all_with_dynamic_tmax()
        tb_tmax.set_val(str(state['T_max']))
        lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")

        txt_msg.set_text(f'Cargado: {os.path.basename(fname)}')
        render()
    except Exception as e:
        txt_msg.set_text(f'Error al cargar: {e}')
        fig.canvas.draw_idle()

btn_load.on_clicked(cargar)


# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────
def reset(ev):
    state['t_max_usuario'] = T_MAX_DEFAULT
    state['T_max']         = T_MAX_DEFAULT
    state['offset_s']      = T_OFFSET_DEFAULT
    state['offset_on']     = 1
    state['auto_tmax']     = True
    btn_auto.label.set_text(_auto_label())
    btn_auto.ax.set_facecolor('#1a4a2a')
    btn_auto.color = '#1a4a2a'; btn_auto.hovercolor = '#2a6a3a'
    tb_off.set_val(str(T_OFFSET_DEFAULT))
    lbl_off_h.set_text(f'= {T_OFFSET_DEFAULT*S2H:.5f} h')
    radio.set_active(1)
    for idx in range(2):
        for row, field in enumerate(FIELDS):
            val = DEFAULTS[idx][field]
            params[idx][field] = val
            textboxes[idx][row].set_val(str(val))
        visible[idx] = True
        btn_vis[idx].label.set_text(f'Ocultar C{idx+1}')

    recalc_all_with_dynamic_tmax()
    tb_tmax.set_val(str(state['T_max']))
    lbl_tmax_h.set_text(f"= {state['T_max']*S2H:.5f} h")

    txt_msg.set_text('')
    render()

btn_rst.on_clicked(reset)

# ─────────────────────────────────────────────────────────────────────────────
plt.show()
