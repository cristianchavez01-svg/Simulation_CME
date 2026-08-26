import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
import matplotlib

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.titleweight': 'normal'})

# ── PARÁMETROS GLOBALES ───────────────────────────────────────────────────────
R_SOL_KM, R_SOL_STR   = 695700, r'$R_\odot$'
DIST_TIERRA_KM = 149597870.7
DIST_TIERRA_RS = DIST_TIERRA_KM / R_SOL_KM
DENSIDAD_FONDO, T_HORAS, FACTOR_ESCALA = 100, 85, 695700
V_VIENTO_SOLAR, VENTANA_SUAV = 400.0, 2
semilla1, semilla2, RETRASO_CME2 = 435, 1962, 25200.0 #7 horas
FACTOR_COMPRESION = 1.15
DMAX_OVERRIDE     = 5.5 # límite superior para escala de colores (log10 de densidad)

N_PUNTOS_OBS = 5
print(f"Puntos de observación: {N_PUNTOS_OBS}")


# ── MORFOLOGÍA COMPARTIDA ─────────────────────────────────────────────────────
class _Morfo:
    def _rfourier(self, th, amp, fase):
        r = np.zeros_like(th, dtype=float)
        for k,(a,ph) in enumerate(zip(amp,fase), start=2):
            r += a*np.cos(k*th+ph)
        return r

    def _fils(self, th):
        f = np.zeros_like(th, dtype=float)
        for ang,a,anc in zip(self.ang_fil, self.amp_fil, self.ancho_fil):
            f += a*np.exp(-((th-ang)**2)/(2*anc**2))
        return f

    def forma(self, th, r_cme, r_ext_ov=None):
        ap  = np.pi*0.56
        ven = np.clip(np.cos(th/ap*(np.pi/2)), 0, 1)**2
        tha = th * np.where(th>=0, self.asimetria, 2-self.asimetria)
        rr  = r_ext_ov if r_ext_ov is not None else r_cme
        rex = (rr*(0.75+0.45*np.cos(tha))
               + rr*self._rfourier(th,self.amp_ext,self.fase_ext)
               + rr*self._fils(th)) * ven
        gro = np.clip(0.28+0.10*np.cos(tha)
                      +self._rfourier(th,self.amp_gros,self.fase_gros), 0.10, 0.50)
        return rex, np.maximum(rex*(1-gro)*ven, r_cme*0.15*ven)

    def _vel_vec_base(self, TH, R, r_cme, v_t, r_ext_ov=None):
        rex,rin = self.forma(TH, r_cme, r_ext_ov)
        mask = (R>rin) & (R<=rex)
        fd   = np.clip((1+np.cos(TH))**2.5, 0, 1)
        vr   = v_t*fd*(1+0.3*np.cos(TH)**2)
        vt   = 0.05*v_t*np.sin(2*TH)*fd
        vm   = np.sqrt(vr**2+vt**2); vm[vm==0]=1
        vrn  = np.where(mask, vr/vm*fd, np.nan)
        vtn  = np.where(mask, vt/vm*fd, np.nan)
        return vrn*np.cos(TH)-vtn*np.sin(TH), vrn*np.sin(TH)+vtn*np.cos(TH), mask


# ── CLASE CME ─────────────────────────────────────────────────────────────────
class CME(_Morfo):
    def __init__(self, nombre, tr, td, ar, ad, v0, x0, R0, semilla, color, t0=0.0):
        self.nombre, self.tr, self.td = nombre, tr, td
        self.ar, self.ad, self.v0, self.x0 = ar, ad, v0, x0
        self.R0, self.semilla, self.color, self.t0 = R0, semilla, color, t0
        rng = np.random.default_rng(seed=semilla)
        N = 7
        self.amp_ext,  self.fase_ext  = rng.uniform(0.03,0.08,N), rng.uniform(0,2*np.pi,N)
        self.amp_gros, self.fase_gros = rng.uniform(0.02,0.07,N), rng.uniform(0,2*np.pi,N)
        self.asimetria = rng.uniform(0.85, 1.15)
        NF = rng.integers(2, 5)
        self.ang_fil   = rng.uniform(-np.pi/2, np.pi/2, NF)
        self.amp_fil   = rng.uniform(0.10, 0.20, NF)
        self.ancho_fil = rng.uniform(0.30, 0.45, NF)

    def aceleracion(self, s):
        return (self.ar*self.ad)/(self.ad*np.exp(-s/self.tr)+self.ar*np.exp(s/self.td))

    def densidad(self, TH, R, r_cme, t, fi=1.0, r_ext_ov=None):
        if t < self.t0:
            return np.full(R.shape, np.nan), np.zeros(R.shape, dtype=bool)
        rex,rin = self.forma(TH, r_cme, r_ext_ov)
        mask = (R>rin) & (R<=rex)
        da   = np.clip((1+np.cos(TH))**5, 0, None)
        dr   = np.exp(-8*(R/(r_cme+0.1)-1)**2)
        dd   = 100/((r_cme/self.R0)**0.5) / np.sqrt(np.maximum(1., (t-self.t0)/600))
        dens = DENSIDAD_FONDO * dd * da * (0.3+dr) * fi
        c    = np.where(mask, dens, np.nan)
        if np.any(mask):
            c = np.where(mask, np.clip(c, DENSIDAD_FONDO, np.nanmax(c)), np.nan)
        return c, mask

    def vel_vec(self, TH, R, r_cme, v_t, r_ext_ov=None):
        return self._vel_vec_base(TH, R, r_cme, v_t, r_ext_ov)


# ── CLASE CME NUEVA ───────────────────────────────────────────────────────────
class CMENueva(_Morfo):
    """
    CME nacida de zona solapada.
    - Morfología fija basada en la geometría del solapamiento al nacer.
    - Se propaga libremente con velocidad heredada.
    - NO vacía densidad de otras CMEs: solo añade la suya.
    - Se restringe únicamente por proximidad de radio al nacer.
    """
    _contador = 0

    def __init__(self, t_nac, r_nac, v_nac, d_nac,
                 theta_centro, apertura_angular, grosor_frac, asimetria):
        CMENueva._contador += 1
        self.id               = CMENueva._contador
        self.t_nac            = t_nac
        self.r_nac            = r_nac
        self.v_nac            = v_nac
        self.d_nac            = d_nac
        self.theta_centro     = theta_centro
        self.apertura_angular = max(apertura_angular, 0.08)
        self.grosor_frac      = np.clip(grosor_frac, 0.12, 0.55)
        self.asimetria        = asimetria

        # Parámetros Fourier nulos: forma viene de apertura angular
        N = 7
        self.amp_ext   = np.zeros(N); self.fase_ext  = np.zeros(N)
        self.amp_gros  = np.zeros(N); self.fase_gros = np.zeros(N)
        self.ang_fil   = np.array([0.0])
        self.amp_fil   = np.array([0.0])
        self.ancho_fil = np.array([1.0])

    def radio(self, t):
        if t < self.t_nac: return None
        return self.r_nac + self.v_nac*(t-self.t_nac)/FACTOR_ESCALA

    def forma(self, th, r_cme, r_ext_ov=None):
        # Ventana angular centrada en theta_centro
        dth = (th - self.theta_centro + np.pi) % (2*np.pi) - np.pi
        ven = np.clip(np.cos(dth/self.apertura_angular*(np.pi/2)), 0, 1)**2
        rex = r_cme * (0.75 + 0.45) * ven
        rin = np.maximum(rex*(1-self.grosor_frac), r_cme*0.10)
        return rex, rin

    def densidad(self, TH, R, t):
        r = self.radio(t)
        if r is None or r <= 0:
            return np.zeros_like(R, dtype=float), np.zeros(R.shape, dtype=bool)
        rex, rin = self.forma(TH, r)
        mask = (R > rin) & (R <= rex)
        dth  = (TH - self.theta_centro + np.pi) % (2*np.pi) - np.pi
        ven  = np.clip(np.cos(dth/self.apertura_angular*(np.pi/2)), 0, 1)**2
        fe   = max(r/self.r_nac, 1e-6)
        # Densidad decae por expansión geométrica 1/r²
        return np.where(mask, self.d_nac/(fe**2)*ven, 0.0), mask

    def vel_vec(self, TH, R, t):
        r = self.radio(t)
        if r is None or r <= 0:
            nan = np.full(R.shape, np.nan)
            return nan, nan, np.zeros(R.shape, dtype=bool)
        rex, rin = self.forma(TH, r)
        mask = (R > rin) & (R <= rex)
        fd   = np.clip((1+np.cos(TH-self.theta_centro))**2.5, 0, 1)
        vr   = self.v_nac*fd*(1+0.3*np.cos(TH)**2)
        vt   = 0.05*self.v_nac*np.sin(2*TH)*fd
        vm   = np.sqrt(vr**2+vt**2); vm[vm==0]=1
        vrn  = np.where(mask, vr/vm*fd, np.nan)
        vtn  = np.where(mask, vt/vm*fd, np.nan)
        return vrn*np.cos(TH)-vtn*np.sin(TH), vrn*np.sin(TH)+vtn*np.cos(TH), mask


# ── FUNCIONES AUXILIARES ──────────────────────────────────────────────────────
def radio_inter(r1,r2,v1,v2,dt=600.):
    return (r1+r2)/2 + (v1+v2)*dt/FACTOR_ESCALA

def fi_inter(v1,v2):
    return ((v1+v2)/max((v1+v2)/2,1.))**0.5

def v_pond(d1, d2, v1, v2):
    d1m = float(np.nanmean(np.nan_to_num(d1))) if np.any(~np.isnan(d1)) else 1.
    d2m = float(np.nanmean(np.nan_to_num(d2))) if np.any(~np.isnan(d2)) else 1.
    return (d1m*v1 + d2m*v2) / max(d1m+d2m, 1e-10)

def densidad_solapada(ci1, ci2, sol):
    d1 = np.nan_to_num(ci1); d2 = np.nan_to_num(ci2)
    total = d1 + d2
    peso  = np.where(total > 0, total, 1.0)
    return np.where(sol, (d1**2 + d2**2)/peso * FACTOR_COMPRESION, 0.0)

def geometria_sol(sol_mask, TH, R, r_ref):
    """Extrae geometría de la zona solapada."""
    if not np.any(sol_mask):
        return 0.0, 0.3, 0.28
    th_s = TH[sol_mask]; r_s = R[sol_mask]
    th_c  = float(np.mean(th_s))
    ap    = max(float(np.ptp(th_s))/2, 0.08)
    grosor = float(np.ptp(r_s)) / max(r_ref, 0.1)
    return th_c, ap, np.clip(grosor, 0.12, 0.55)

def umbral(r):
    return max(r*0.12, 0.15)

def ya_existe(r_nuevo, lista_a, lista_b):
    return any(abs(cn.r_nac-r_nuevo) < umbral(r_nuevo)
               for cn in lista_a+lista_b)

def calc_campos(ca, cb, TH, R, t, ra, rb, va, vb, ri, fi):
    c1,m1 = ca.densidad(TH,R,ra,t)
    c2,m2 = cb.densidad(TH,R,rb,t)
    sol=m1&m2; sa=m1&~m2; sb=~m1&m2
    if np.any(sol):
        ci1,_ = ca.densidad(TH,R,ra,t,fi=fi,r_ext_ov=ri)
        ci2,_ = cb.densidad(TH,R,rb,t,fi=fi,r_ext_ov=ri)
        ds = densidad_solapada(ci1, ci2, sol)
    else:
        ci1=ci2=np.full(R.shape,np.nan); ds=np.zeros(R.shape)
    d = np.where(sa,np.nan_to_num(c1),np.where(sb,np.nan_to_num(c2),np.where(sol,ds,0.)))
    return d, m1, m2, sol, c1, c2, ci1, ci2


def detectar_y_registrar(sol_mask, campo_a, campo_b, v_a, v_b,
                          r_nuevo, asim, TH, R, t,
                          cmes_nuevas, nuevas_frame):
    """
    Si hay solapamiento, detecta geometría y registra nueva CME.
    No vacía ni modifica df — solo registra.
    """
    if not np.any(sol_mask): return
    if ya_existe(r_nuevo, cmes_nuevas, nuevas_frame): return

    th_c, ap, gro = geometria_sol(sol_mask, TH, R, r_nuevo)
    d1 = np.nan_to_num(campo_a); d2 = np.nan_to_num(campo_b)
    total = d1 + d2
    peso  = np.where(total > 0, total, 1.0)
    ds_zona = np.where(sol_mask, (d1**2+d2**2)/peso*FACTOR_COMPRESION, 0.0)
    ds = float(np.nanmean(ds_zona[sol_mask])) if np.any(sol_mask) else 0.
    if ds <= 0: return

    vs = v_pond(campo_a, campo_b, v_a, v_b)
    nuevas_frame.append(CMENueva(t, r_nuevo, vs, ds, th_c, ap, gro, asim))


def reset_simulation_state():
    """Reinicia las variables globales para evitar contaminación entre ejecuciones."""
    CMENueva._contador = 0
    globals()["cmes_nuevas"] = []


# ── INSTANCIAS ────────────────────────────────────────────────────────────────
reset_simulation_state()

cme1 = CME('CME-1',tr=6900,td=55600,ar=0.034,ad=0.01,v0=100,x0=100000,R0=5.2,
           semilla=semilla1,color='steelblue',t0=0.0)
cme2 = CME('CME-2',tr=3500,td=4651.77,ar=0.03, ad=0.17,v0=140,x0=140000,R0=4.0,
           semilla=semilla2,color='red',t0=RETRASO_CME2)
cmes_nuevas = []


# ── CINEMÁTICA ────────────────────────────────────────────────────────────────
tiempos   = np.linspace(0, T_HORAS*3600, 500)
tiempos_h = tiempos/3600

def cinematica(cme, t_arr):
    s   = t_arr - cme.t0; act = s >= 0
    ac  = np.where(act, [cme.aceleracion(max(0.,si)) for si in s], np.nan)
    vel = np.where(act, cme.v0+cumulative_trapezoid(np.where(act,ac,0.),t_arr,initial=0.), np.nan)
    pos = np.where(act, cme.x0+cumulative_trapezoid(np.where(act,vel,0.),t_arr,initial=0.), np.nan)
    return pos, vel, ac*1000.

print("Calculando cinemática...")
pos1,vel1,acel1 = cinematica(cme1, tiempos)
pos2,vel2,acel2 = cinematica(cme2, tiempos)
pos1 = np.where(np.isnan(pos1), cme1.x0, pos1)
pos1_rs, pos2_rs = pos1/R_SOL_KM, pos2/R_SOL_KM

def primera_interseccion(t_h, pos_a, pos_b):
    validos = ~np.isnan(pos_a) & ~np.isnan(pos_b)
    t_validos = t_h[validos]
    diferencia = pos_a[validos] - pos_b[validos]
    if len(diferencia) == 0:
        return None, None

    exactos = np.flatnonzero(diferencia == 0)
    if len(exactos):
        i = exactos[0]
        return t_validos[i], pos_a[validos][i]

    cruces = np.flatnonzero(diferencia[:-1] * diferencia[1:] < 0)
    if len(cruces) == 0:
        return None, None
    i = cruces[0]
    fraccion = -diferencia[i] / (diferencia[i + 1] - diferencia[i])
    t_inter = t_validos[i] + fraccion * (t_validos[i + 1] - t_validos[i])
    pos_inter = pos_a[validos][i] + fraccion * (pos_a[validos][i + 1] - pos_a[validos][i])
    return t_inter, pos_inter

t_centros, pos_centros = primera_interseccion(tiempos_h, pos1_rs, pos2_rs)

def etapas(ac, vel, th):
    return th[np.nanargmax(ac)], th[np.argmax(np.nan_to_num(vel)>=0.95*np.nanmax(vel))]

t_inic1,t_acel1 = etapas(acel1,vel1,tiempos_h)
t_inic2,t_acel2 = etapas(acel2,vel2,tiempos_h)
print(f"CME-1: ini={t_inic1:.2f}h  acel={t_acel1:.2f}h")
print(f"CME-2: ini={t_inic2:.2f}h  acel={t_acel2:.2f}h")

_idx_cache = {}
def _idx(t):
    if t not in _idx_cache: _idx_cache[t]=int(np.argmin(np.abs(tiempos-t)))
    return _idx_cache[t]

def pos_rs_en(cme, t):
    p = pos1[_idx(t)] if cme is cme1 else pos2[_idx(t)]
    return None if np.isnan(p) else p/R_SOL_KM

def vel_en(cme, t):
    v = vel1[_idx(t)] if cme is cme1 else vel2[_idx(t)]
    return 0. if np.isnan(v) else float(v)


# ── FRENTE Y RETAGUARDIA ──────────────────────────────────────────────────────
def frente_retaguardia(cme, pos_arr):
    rex_arr = np.full_like(pos_arr, np.nan)
    rin_arr = np.full_like(pos_arr, np.nan)
    th0 = np.array([0.0])
    for i,p in enumerate(pos_arr):
        if not np.isnan(p):
            rex,rin = cme.forma(th0, p/R_SOL_KM)
            rex_arr[i], rin_arr[i] = float(rex[0]), float(rin[0])
    return rex_arr, rin_arr

print("Calculando frente y retaguardia...")
rex1,rin1 = frente_retaguardia(cme1, pos1)
rex2,rin2 = frente_retaguardia(cme2, pos2)
t_extensiones, pos_extensiones = primera_interseccion(tiempos_h, rex2, rin1)


# ── 1. CINEMÁTICA CONJUNTA ────────────────────────────────────────────────────
def sombrear(ax, ti1, ta1, ti2, ta2):
    ax.axvspan(0, T_HORAS, color='#FFF', alpha=1., zorder=0)
    ax.axvspan(ti1, ta1, color='#A9C5E3', alpha=.25, zorder=0)
    ax.axvspan(ti2, ta2, color='#E3B57A', alpha=.25, zorder=0)
    for x, ls in [(ti1, '--'), (ta1, ':'), (ti2, '--'), (ta2, ':')]:
        ax.axvline(x=x, color='black', ls=ls, lw=.8, alpha=.5, zorder=2)

def dibujar_linea_tiempo(fig, ax_ref, ti1, ta1, ti2, ta2):
    """Dibuja las fases fuera de los ejes de datos, como una línea de tiempo."""
    posicion = ax_ref.get_position()
    timeline = fig.add_axes([posicion.x0, .80, posicion.width, .055])
    timeline.set_xlim(0, T_HORAS)
    timeline.set_ylim(-.45, 1.45)
    timeline.set_yticks([1, 0], ['CME-1', 'CME-2'])
    timeline.set_xticks([])
    timeline.tick_params(axis='both', length=0, labelsize=10, colors='black')
    timeline.spines[:].set_visible(False)

    fases = [
        (1, [(0, ti1, 'Ini'), (ti1, ta1, 'Acel'), (ta1, T_HORAS, 'Prop')]),
        (0, [(cme2.t0/3600, ti2, 'Ini'), (ti2, ta2, 'Acel'),
             (ta2, T_HORAS, 'Prop')])]
    for y, tramos in fases:
        inicio_linea = tramos[0][0]
        fin_linea = tramos[-1][1]
        timeline.plot([inicio_linea, fin_linea], [y, y], color='black', lw=1.8,
                      solid_capstyle='butt', zorder=2)
        timeline.vlines([inicio_linea, fin_linea], y - .24, y + .24,
                color='black', lw=1.2, zorder=3)
        for inicio, fin, nombre in tramos:
            if fin <= inicio:
                continue
            if inicio > inicio_linea:
                timeline.vlines(inicio, y - .24, y + .24, color='black', lw=1.2, zorder=3)
            timeline.text((inicio + fin) / 2, y + (.22 if y == 1 else -.22), nombre,
                          ha='center', va='center', fontsize=9, color='black', zorder=3)

fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True,gridspec_kw={'hspace':0})
fig.subplots_adjust(top=0.76,hspace=0.18)
fig.suptitle('Cinemática conjunta: CME-1 y CME-2',fontsize=20,y=.985)
subtitle = (
    rf'$\mathrm{{CME}}_1:\ a_r={cme1.ar:.2f},\ a_d={cme1.ad:.2f},\ \tau_r={cme1.tr:.0f},\ \tau_d={cme1.td:.0f}$'
    + '\n'
    rf'$\mathrm{{CME}}_2:\ a_r={cme2.ar:.2f},\ a_d={cme2.ad:.2f},\ \tau_r={cme2.tr:.0f},\ \tau_d={cme2.td:.0f}$'
)
fig.text(.5,.935,subtitle,ha='center',va='top',fontsize=11,style='italic',color='#444',linespacing=1.25)
fig.text(.5,.875,f'{T_HORAS} horas de propagación',ha='center',fontsize=12,style='italic',color='#444')
dibujar_linea_tiempo(fig, axes[0], t_inic1, t_acel1, t_inic2, t_acel2)
ax_a,ax_v,ax_p = axes; kw=dict(linewidth=2.5,zorder=3)
for ax in axes:
    sombrear(ax, t_inic1, t_acel1, t_inic2, t_acel2)
ax_a.plot(tiempos_h,acel1,color=cme1.color,**kw); ax_a.plot(tiempos_h,acel2,color=cme2.color,**kw)
ax_a.axhline(0,color='k',ls='-',alpha=.3,lw=.5,zorder=2)
ax_a.set_ylabel(r'Aceleración (m/s$^2$)',fontsize=16)
ax_v.plot(tiempos_h,vel1,color=cme1.color,**kw); ax_v.plot(tiempos_h,vel2,color=cme2.color,**kw)
ax_v.set_ylabel('Velocidad (km/s)',fontsize=16)
ax_p.plot(tiempos_h,pos1_rs,color=cme1.color,label='CME-1',**kw)
ax_p.plot(tiempos_h,pos2_rs,color=cme2.color,label='CME-2',**kw)
ax_p.fill_between(tiempos_h,rin1,rex1,color=cme1.color,alpha=.15,label='Extensión CME-1')
ax_p.fill_between(tiempos_h,rin2,rex2,color=cme2.color,alpha=.15,label='Extensión CME-2')
if t_centros is not None:
    ax_p.scatter(t_centros, pos_centros, marker='o', s=110, color='white',
                 edgecolors='black', linewidths=1.4, zorder=6,
                 label='Interacción de centros')
if t_extensiones is not None:
    ax_p.scatter(t_extensiones, pos_extensiones, marker='X', s=130, color='black',
                 edgecolors='white', linewidths=1.2, zorder=6,
                 label='Interacción de extensiones')
ax_p.set_ylabel(f'Posición ({R_SOL_STR})',fontsize=16)
ax_p.set_xlabel('Tiempo (h)',fontsize=16)
major_ticks = np.arange(0,T_HORAS+1,5)
minor_ticks = np.arange(0,T_HORAS+1,1)
for ax in axes:
    ax.set_xlim(0,T_HORAS)
    ax.grid(True,alpha=.3,ls='--',zorder=1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', which='major', length=10, width=1.4, direction='in')
    ax.tick_params(axis='x', which='minor', length=5, width=0.8, direction='in')
    ax.tick_params(axis='y', which='major', direction='in', right=True)
    ax.tick_params(axis='y', which='minor', direction='in', right=True)
    ax.yaxis.set_ticks_position('both')
    ax.grid(which='minor', axis='x', alpha=0.15, ls='--')
    ax.grid(which='major', axis='x', alpha=0.35, ls='--')
ax_a.tick_params(axis='x', which='both', labelbottom=False, bottom=False, top=False)
ax_v.tick_params(axis='x', which='both', labelbottom=False, bottom=False, top=False)
ax_p.set_xticks(major_ticks)
ax_p.set_xticklabels([str(int(x)) for x in major_ticks])
ax_p.axhline(y=DIST_TIERRA_RS, color='k', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
ax_p.text(T_HORAS*0.02, DIST_TIERRA_RS*1.02, 'Tierra', color='k', fontsize=11, va='bottom', ha='left')
ax_a.legend(*ax_p.get_legend_handles_labels(),loc='upper right',fontsize=11)
plt.savefig(f"cinematica_conjunta_s1_{semilla1}_s2_{semilla2}.pdf",
            dpi=300,bbox_inches='tight',pad_inches=0.3)
print("✓ Cinemática guardada"); plt.show()


# ── 2. PROPAGACIÓN POLAR ──────────────────────────────────────────────────────
print("\n"+"="*60+"\nVISUALIZACIÓN POLAR\n"+"="*60)
ESTADOS  = 8
t_frames = np.linspace(3600, T_HORAS*3600, ESTADOS)
RMAX = max(pos1[-1], pos2[~np.isnan(pos2)][-1]) / R_SOL_KM * 1.4

TH_G,R_G   = np.meshgrid(np.linspace(-np.pi,np.pi,800), np.linspace(0,RMAX,400))
THv_G,Rv_G = np.meshgrid(np.linspace(-np.pi,np.pi,40),  np.linspace(1,RMAX,12))

print("  Pre-calculando rango densidad...")
dmax = DENSIDAD_FONDO
for t_pre in t_frames:
    r1=pos_rs_en(cme1,t_pre); v1=vel_en(cme1,t_pre)
    if t_pre>=cme2.t0 and pos_rs_en(cme2,t_pre):
        r2=pos_rs_en(cme2,t_pre); v2=vel_en(cme2,t_pre)
        fi=fi_inter(v1,v2); ri=radio_inter(r1,r2,v1,v2)
        d,*_=calc_campos(cme1,cme2,TH_G,R_G,t_pre,r1,r2,v1,v2,ri,fi)
    else:
        c1,m1=cme1.densidad(TH_G,R_G,r1,t_pre); d=np.where(m1,np.nan_to_num(c1),0.)
    # CMEs nuevas solo añaden, no vacían
    for cn in cmes_nuevas:
        dcn,_=cn.densidad(TH_G,R_G,t_pre); d=d+dcn
    dmax=max(dmax, float(np.nanmax(d)) if np.any(d>0) else DENSIDAD_FONDO)

DMIN=float(np.log10(DENSIDAD_FONDO))
DMAX=float(DMAX_OVERRIDE) if DMAX_OVERRIDE else max(float(np.log10(dmax)),DMIN+0.5)
print(f"  Rango: [{DMIN:.2f}, {DMAX:.2f}]")

fig=plt.figure(figsize=(20,12))
fig.suptitle('Propagación conjunta: CME-1 y CME-2',fontsize=18,fontweight='normal',y=.99)
fig.text(.5,.935,f'{T_HORAS} horas de propagación',ha='center',fontsize=13,style='italic',color='#444')

for idx,t_fr in enumerate(t_frames):
    print(f"  Frame {idx+1}/{ESTADOS}: t={t_fr/3600:.1f}h",end=" ... ")
    ax=plt.subplot(2,4,idx+1,projection='polar')
    ax.text(.05,.97,f"{'abcdefgh'[idx]})",transform=ax.transAxes,fontsize=12,
            va='top',ha='left',color='black',fontstyle='italic')

    r1=pos_rs_en(cme1,t_fr); v1=vel_en(cme1,t_fr)
    act2=t_fr>=cme2.t0 and pos_rs_en(cme2,t_fr) is not None
    if act2: r2=pos_rs_en(cme2,t_fr); v2=vel_en(cme2,t_fr); fi=fi_inter(v1,v2); ri=radio_inter(r1,r2,v1,v2)
    else:    r2=v2=0.; fi=1.; ri=None

    if idx < 4:
        r_refs=[r1,r2 if act2 else 0,ri if ri else 0]+[cn.radio(t_fr) or 0 for cn in cmes_nuevas]
        rl=max(max(r_refs)*2.7, 1.)
        TH_L,R_L   = np.meshgrid(np.linspace(-np.pi,np.pi,800), np.linspace(0,rl,400))
        THv_L,Rv_L = np.meshgrid(np.linspace(-np.pi,np.pi,40),  np.linspace(1,rl,12))
    else:
        TH_L,R_L,THv_L,Rv_L = TH_G,R_G,THv_G,Rv_G

    ax.contourf(TH_L,R_L,np.ones_like(R_L),levels=[.5,1.5],colors=['#D6EAF8'],alpha=.8)

    # ── Densidad base de CMEs originales ──────────────────────────────────────
    if act2:
        df,m1f,m2f,sol,c1f,c2f,ci1,ci2 = calc_campos(cme1,cme2,TH_L,R_L,t_fr,r1,r2,v1,v2,ri,fi)
        vs = v_pond(ci1,ci2,v1,v2)
    else:
        c1f,m1f=cme1.densidad(TH_L,R_L,r1,t_fr)
        m2f=sol=np.zeros(R_L.shape,dtype=bool)
        ci1=ci2=np.full(R_L.shape,np.nan)
        df=np.where(m1f,np.nan_to_num(c1f),0.); vs=v1

    # ── CMEs nuevas: solo suman su densidad, sin vaciar nada ─────────────────
    mask_cn  = np.zeros(R_L.shape, dtype=bool)
    vel_cn_map = np.zeros(R_L.shape)
    for cn in cmes_nuevas:
        dcn, mcn = cn.densidad(TH_L, R_L, t_fr)
        df = df + dcn                                      # solo suma
        vel_cn_map = np.where(mcn & ~mask_cn, cn.v_nac, vel_cn_map)
        mask_cn = mask_cn | mcn

    # ── Detectar solapamientos y registrar nuevas CMEs ────────────────────────
    nuevas_frame = []
    if act2:
        # Pre-calcular campos de CMEs nuevas existentes una sola vez
        campos_cn = [(cn, *cn.densidad(TH_L,R_L,t_fr)) for cn in cmes_nuevas]

        # 1) CME1 vs CME2 originales
        detectar_y_registrar(m1f&m2f, ci1, ci2, v1, v2,
                              ri or 0., cme1.asimetria,
                              TH_L, R_L, t_fr, cmes_nuevas, nuevas_frame)

        # 2) CMEs originales vs CMEs nuevas
        for cn,dcn,mcn in campos_cn:
            r_cn = cn.radio(t_fr)
            if r_cn is None: continue
            # CME1 vs nueva
            detectar_y_registrar(m1f&mcn, ci1, dcn, v1, cn.v_nac,
                                  (r_cn+(r1 or r_cn))/2, cme1.asimetria,
                                  TH_L, R_L, t_fr, cmes_nuevas, nuevas_frame)
            # CME2 vs nueva
            detectar_y_registrar(m2f&mcn, ci2, dcn, v2, cn.v_nac,
                                  (r_cn+(r2 or r_cn))/2, cme2.asimetria,
                                  TH_L, R_L, t_fr, cmes_nuevas, nuevas_frame)

        # 3) CMEs nuevas entre sí
        for i,(cn_a,da,ma) in enumerate(campos_cn):
            for cn_b,db,mb in campos_cn[i+1:]:
                ra=cn_a.radio(t_fr); rb=cn_b.radio(t_fr)
                if ra is None or rb is None: continue
                detectar_y_registrar(ma&mb, da, db, cn_a.v_nac, cn_b.v_nac,
                                      (ra+rb)/2, cn_a.asimetria,
                                      TH_L, R_L, t_fr, cmes_nuevas, nuevas_frame)

    if nuevas_frame:
        print(f"\n    ★ {len(nuevas_frame)} nueva(s) t={t_fr/3600:.2f}h", end=" ")
        cmes_nuevas.extend(nuevas_frame)
        # Añadir densidad de las recién nacidas (solo suma)
        for cn in nuevas_frame:
            dcn,mcn = cn.densidad(TH_L,R_L,t_fr)
            df = df + dcn
            vel_cn_map = np.where(mcn & ~mask_cn, cn.v_nac, vel_cn_map)
            mask_cn = mask_cn | mcn

    ax.contourf(TH_L,R_L,np.log10(np.where(df>0,df,DENSIDAD_FONDO/10)),
                levels=np.linspace(DMIN,DMAX,100),cmap='viridis',alpha=.9,extend='max')

    # ── Vectores ──────────────────────────────────────────────────────────────
    U1,V1,mv1=cme1.vel_vec(THv_L,Rv_L,r1,v1)
    if act2:
        U2,V2,mv2=cme2.vel_vec(THv_L,Rv_L,r2,v2)
        Ui1,Vi1,_=cme1.vel_vec(THv_L,Rv_L,r1,vs,r_ext_ov=ri)
        Ui2,Vi2,_=cme2.vel_vec(THv_L,Rv_L,r2,vs,r_ext_ov=ri)
    else:
        U2=V2=np.full(THv_L.shape,np.nan); mv2=np.zeros(THv_L.shape,dtype=bool)
        Ui1,Vi1,Ui2,Vi2=U1,V1,U2,V2

    sv1=mv1&~mv2; sv2=~mv1&mv2; svs=mv1&mv2
    Up=np.where(sv1,U1,np.where(sv2,U2,np.where(svs,Ui1+Ui2,np.nan)))
    Vp=np.where(sv1,V1,np.where(sv2,V2,np.where(svs,Vi1+Vi2,np.nan)))
    for cn in cmes_nuevas:
        Ucn,Vcn,mvcn=cn.vel_vec(THv_L,Rv_L,t_fr)
        Up=np.where(mvcn,Ucn,Up); Vp=np.where(mvcn,Vcn,Vp)

    mag=np.sqrt(np.nan_to_num(Up)**2+np.nan_to_num(Vp)**2)
    mm=float(np.nanmax(mag)) if np.any(~np.isnan(Up)) else 1.
    if mm>0: Up,Vp=Up/mm*.3,Vp/mm*.3
    ax.quiver(THv_L,Rv_L,Up,Vp,scale=8,width=.002,headwidth=2,
              headlength=2,headaxislength=2.5,color='red',alpha=.9)

    tit=(f"t={t_fr/3600:.1f}h  r₁={r1:.1f}  r₂={r2:.1f} {R_SOL_STR}  "
         f"v_sol={vs:.0f} km/s  N={len(cmes_nuevas)}"
         if act2 else f"t={t_fr/3600:.1f}h  r₁={r1:.1f} {R_SOL_STR}  v={v1:.0f} km/s")
    ax.set_title(tit,fontsize=9,fontweight='normal',pad=10)
    r_refs=[r1,r2 if act2 else 0,ri if ri else 0]+[cn.radio(t_fr) or 0 for cn in cmes_nuevas]
    ax.set_ylim([0,max(max(r_refs)*2.7,1.) if idx<4 else RMAX])
    ax.set_rlabel_position(135); ax.grid(True,alpha=.3,ls='--',lw=.7)
    print(f"r₁={r1:.2f}"+(f"  r₂={r2:.2f}  N={len(cmes_nuevas)}" if act2 else "")+" ✓")

plt.tight_layout(rect=[0,0,.92,.97])
cbar_ax=fig.add_axes([.94,.12,.015,.75])
sm=plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(vmin=DMIN,vmax=DMAX)); sm.set_array([])
fig.colorbar(sm,cax=cbar_ax).set_label(r'log$_{10}$($\rho$) [protones/cm$^3$]',
                                         rotation=270,labelpad=25,fontsize=11)
plt.savefig(f"cme_conjunta_polar_s1_{semilla1}_s2_{semilla2}.pdf",dpi=300,bbox_inches='tight')
print(f"\n✓ Polar guardado | CMEs nuevas: {len(cmes_nuevas)}"); plt.show()


# ── 3. SERIES TEMPORALES ──────────────────────────────────────────────────────
PUNTOS_OBS = [(round(r,4), 0.0) for r in np.linspace(0, RMAX, N_PUNTOS_OBS)]
print("\n"+"="*60+f"\nSERIES TEMPORALES — {len(PUNTOS_OBS)} puntos\n"+"="*60)
th_loc=np.linspace(-np.pi,np.pi,200); r_loc=np.linspace(0,RMAX,250)
TH_loc,R_loc=np.meshgrid(th_loc,r_loc)
dens_ser={p: np.full(len(tiempos),DENSIDAD_FONDO) for p in PUNTOS_OBS}
vel_ser ={p: np.full(len(tiempos),V_VIENTO_SOLAR)  for p in PUNTOS_OBS}
idx_pts ={(ro,to): (np.argmin(np.abs(r_loc-ro)), np.argmin(np.abs(th_loc-np.radians(to))))
          for ro,to in PUNTOS_OBS}

print("  Calculando series temporales...")
for i,t in enumerate(tiempos):
    if i%50==0: print(f"    t={t/3600:.2f}h",end="... ")
    r1t=pos_rs_en(cme1,t); v1t=vel_en(cme1,t)
    act2=t>=cme2.t0 and pos_rs_en(cme2,t) is not None
    if act2:
        r2t=pos_rs_en(cme2,t); v2t=vel_en(cme2,t)
        fi_t=fi_inter(v1t,v2t); ri_t=radio_inter(r1t,r2t,v1t,v2t)
        dt,m1t,m2t,sol_t,_,_,ci1t,ci2t=calc_campos(cme1,cme2,TH_loc,R_loc,t,
                                                      r1t,r2t,v1t,v2t,ri_t,fi_t)
        vs_t=v_pond(ci1t,ci2t,v1t,v2t)
    else:
        c1t,m1t=cme1.densidad(TH_loc,R_loc,r1t,t)
        m2t=sol_t=np.zeros(R_loc.shape,dtype=bool)
        dt=np.where(m1t,np.nan_to_num(c1t),0.); v2t=vs_t=0.

    # CMEs nuevas: solo suma, sin vaciar
    mask_cn=np.zeros(R_loc.shape,dtype=bool)
    vel_cn_map=np.zeros(R_loc.shape)
    for cn in cmes_nuevas:
        dcn,mcn=cn.densidad(TH_loc,R_loc,t)
        dt=dt+dcn
        vel_cn_map=np.where(mcn&~mask_cn, cn.v_nac, vel_cn_map)
        mask_cn=mask_cn|mcn

    for ro,to in PUNTOS_OBS:
        ir,it_=idx_pts[(ro,to)]; dv=dt[ir,it_]
        if dv>0: dens_ser[(ro,to)][i]=dv
        if   mask_cn[ir,it_]:       vel_ser[(ro,to)][i]=vel_cn_map[ir,it_]
        elif sol_t[ir,it_]:         vel_ser[(ro,to)][i]=vs_t
        elif m1t[ir,it_]:           vel_ser[(ro,to)][i]=v1t
        elif act2 and m2t[ir,it_]:  vel_ser[(ro,to)][i]=v2t
    if i%50==0: print("✓")
print("  ✓ Series calculadas")

CMAP_OBS='gist_rainbow'; cmap_obs=plt.cm.get_cmap(CMAP_OBS)
norm_obs=plt.Normalize(vmin=min(r for r,_ in PUNTOS_OBS),vmax=max(r for r,_ in PUNTOS_OBS))
fig,(ax_d,ax_v)=plt.subplots(2,1,figsize=(14,9),sharex=True,gridspec_kw={'hspace':0})
fig.suptitle(rf'Evolución temporal — {len(PUNTOS_OBS)} puntos',fontsize=15,fontweight='normal',y=.98)
fig.text(.5,.945,f'{T_HORAS} horas de propagación',ha='center',fontsize=12,style='italic',color='#444')
for ro,to in PUNTOS_OBS:
    c=cmap_obs(norm_obs(ro)); k=(ro,to)
    ax_d.plot(tiempos_h,gaussian_filter1d(dens_ser[k],sigma=VENTANA_SUAV),color=c,lw=1.8,alpha=.6,zorder=3)
    ax_v.plot(tiempos_h,gaussian_filter1d(vel_ser[k], sigma=VENTANA_SUAV),color=c,lw=1.8,alpha=.6,zorder=3)
kwr=dict(lw=1.,alpha=.8,zorder=4)
ax_d.axhline(DENSIDAD_FONDO,color='gray',ls='--',label='Fondo',**kwr)
ax_d.axvline(cme2.t0/3600,color='black',ls=':',label='Inicio CME-2',**kwr)
ax_v.axhline(V_VIENTO_SOLAR,color='gray',ls='--',label=f'Viento solar ({V_VIENTO_SOLAR:.0f} km/s)',**kwr)
ax_v.axvline(cme2.t0/3600,color='black',ls=':',label='Inicio CME-2',**kwr)
ax_d.set(ylabel='Densidad (protones/cm³)',yscale='log',xlim=(0,T_HORAS))
ax_d.grid(True,alpha=.3,ls='--',zorder=1)
ax_d.legend(fontsize=10,loc='upper right')
ax_v.set(ylabel='Velocidad radial (km/s)',xlabel='Tiempo (h)',
         xlim=(0,T_HORAS),ylim=(V_VIENTO_SOLAR*.9,None))
major_ticks = np.arange(0, T_HORAS+1, 5)
minor_ticks = np.arange(0, T_HORAS+1, 1)
for ax in (ax_d, ax_v):
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(int(x)) for x in major_ticks])
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', which='major', length=10, width=1.4, direction='in')
    ax.tick_params(axis='x', which='minor', length=5, width=0.8, direction='in')
    ax.tick_params(axis='y', which='major', direction='in', right=True)
    ax.tick_params(axis='y', which='minor', direction='in', right=True)
    ax.yaxis.set_ticks_position('both')
    ax.grid(which='minor', axis='x', alpha=0.15, ls='--')
    ax.grid(which='major', axis='x', alpha=0.35, ls='--')
    ax.grid(which='major', axis='y', alpha=0.15, ls='--')
ax_v.legend(fontsize=10,loc='upper right')
sm_o=plt.cm.ScalarMappable(cmap=CMAP_OBS,norm=norm_obs); sm_o.set_array([])
fig.colorbar(sm_o,ax=[ax_d,ax_v],orientation='vertical',fraction=.02,pad=.02).set_label(
    f'Distancia al Sol ({R_SOL_STR})',rotation=270,labelpad=20,fontsize=12)
plt.savefig(f"serie_temporal_multipunto_s1_{semilla1}_s2_{semilla2}.pdf",dpi=300,bbox_inches='tight')
print("✓ Serie temporal guardada"); plt.show()


# ── 4. RESUMEN FINAL ──────────────────────────────────────────────────────────
print("\n"+"="*60+"\nRESUMEN FINAL\n"+"="*60)
for cme,vl,al,pl in [(cme1,vel1,acel1,pos1),(cme2,vel2,acel2,pos2)]:
    pf=pl[~np.isnan(pl)][-1]
    print(f"\n  {cme.nombre} (t={cme.t0/3600:.1f}h) | semilla={cme.semilla} | R0={cme.R0:.1f} R☉")
    print(f"    v0={cme.v0:.1f}  vmax={np.nanmax(vl):.1f}  vf={vl[~np.isnan(vl)][-1]:.1f} km/s")
    print(f"    amax={np.nanmax(al):.4f} m/s²  |  {cme.x0/R_SOL_KM:.3f}→{pf/R_SOL_KM:.3f} R☉")
if cmes_nuevas:
    print(f"\n  CMEs nuevas: {len(cmes_nuevas)} | FACTOR_COMPRESION={FACTOR_COMPRESION}")
    for j,cn in enumerate(cmes_nuevas):
        rf=cn.radio(tiempos[-1])
        print(f"    [N{cn.id}] t={cn.t_nac/3600:.2f}h  r={cn.r_nac:.2f}→{rf:.2f} R☉  "
              f"v={cn.v_nac:.1f} km/s  θ={np.degrees(cn.theta_centro):.1f}°  "
              f"ap={np.degrees(cn.apertura_angular):.1f}°  d={cn.d_nac:.1f}")
print(f"\n  Puntos obs: {len(PUNTOS_OBS)}")
print("="*60+"\n✓ SIMULACIÓN COMPLETADA")