import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import csv
import os
from scipy.optimize import curve_fit

# ------------------------------------------------------------------
# Configuración de gráficos estilo LaTeX
# ------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})
PLOT_DPI = 200
PLOT_POINTS = 200

# ------------------------------------------------------------------
# 1. DATOS DE LA TABLA
# ------------------------------------------------------------------
a_d = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17])

t_d = np.array([57050, 26346, 17418, 13721, 10837, 9305, 8238, 7450,
                6844, 6362, 5968, 5640, 5362, 5123, 4916, 4733, 4571])

Vmax95 = np.array([663.66, 607.26, 583.5, 564.9, 554.41, 547.52, 542.52,
                    538.21, 535.17, 533.02, 531.13, 529.71, 527.89, 526.73,
                    525.36, 524.78, 524.03])

T_interaccion = np.array([89.49, 78.6, 45.0, 32.6, 34.2, 32.66, 31.29,
                           30.77, 29.69, 29.0, 28.75, 28.46, 28.27, 28.18,
                           27.55, 27.56, 27.43])

Altura_interaccion = np.array([1.17, 1.01, 0.50, 0.39, 0.34, 0.32, 0.30,
                                0.29, 0.28, 0.27, 0.27, 0.266, 0.264, 0.263,
                                0.254, 0.255, 0.253])

T95_Vmin = np.array([49.4, 27.5, 20.7, 17.57, 15.8, 14.69, 13.92, 13.3,
                      12.86, 12.5, 12.26, 12.03, 11.81, 11.65, 11.48,
                      11.37, 11.26])

OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. MODELOS DE AJUSTE
# ------------------------------------------------------------------

def r2_score(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - ss_res / ss_tot


def ajustar_lineal(x, y):
    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b
    r2 = r2_score(y, y_pred)
    eq = f"$y = {a:.6g}\\,x + {b:.6g}$"
    return dict(nombre="Lineal", params=(a, b), r2=r2, eq=eq,
                func=lambda xv, a=a, b=b: a * xv + b)


def ajustar_potencial(x, y):
    if np.any(x <= 0) or np.any(y <= 0):
        return None
    
    def potencial(x, a, b):
        with np.errstate(over='ignore', invalid='ignore'):
            return a * np.power(x, b)
    
    try:
        popt, _ = curve_fit(potencial, x, y, p0=[np.max(y), -1], maxfev=100000)
        a, b = popt
    except Exception:
        # Fallback a polyfit si curve_fit falla
        b, log_a = np.polyfit(np.log(x), np.log(y), 1)
        a = np.exp(log_a)
    
    y_pred = a * np.power(x, b)
    r2 = r2_score(y, y_pred)
    eq = f"$y = {a:.6g}\\,x^{{{b:.6g}}}$"
    
    def func_potencial(xv, a=a, b=b):
        with np.errstate(over='ignore', invalid='ignore'):
            result = a * np.power(xv, b)
            result = np.where(np.isfinite(result), result, np.nan)
        return result
    
    return dict(nombre="Potencial", params=(a, b), r2=r2, eq=eq,
                func=func_potencial)


def ajustar_exponencial(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return None

    def modelo(x, a, b):
        with np.errstate(over='ignore', invalid='ignore'):
            return a * np.exp(b * x)

    try:
        # Estimar valores iniciales con polyfit en log
        b_init, log_a_init = np.polyfit(x, np.log(y), 1)
        a_init = np.exp(log_a_init)
        params, _ = curve_fit(modelo, x, y, p0=[a_init, b_init], maxfev=100000)
        a, b = params
    except Exception:
        # Fallback a polyfit si curve_fit falla
        b, log_a = np.polyfit(x, np.log(y), 1)
        a = np.exp(log_a)

    y_pred = a * np.exp(b * x)
    r2 = r2_score(y, y_pred)
    eq = f"$y = {a:.6g}\\,e^{{{b:.6g}\\,x}}$"
    
    def func_exponencial(xv, a=a, b=b):
        with np.errstate(over='ignore', invalid='ignore'):
            result = a * np.exp(b * xv)
            result = np.where(np.isfinite(result), result, np.nan)
        return result
    
    return dict(nombre="Exponencial", params=(a, b), r2=r2, eq=eq,
                func=func_exponencial)


def ajustar_logaritmico(x, y):
    if np.any(x <= 0):
        return None
    
    def modelo(x, a, b):
        return a * np.log(x) + b
    
    try:
        # Estimar valores iniciales con polyfit
        a_init, b_init = np.polyfit(np.log(x), y, 1)
        params, _ = curve_fit(modelo, x, y, p0=[a_init, b_init], maxfev=100000)
        a, b = params
    except Exception:
        # Fallback a polyfit si curve_fit falla
        a, b = np.polyfit(np.log(x), y, 1)
    
    y_pred = a * np.log(x) + b
    r2 = r2_score(y, y_pred)
    eq = f"$y = {a:.6g}\\,\\ln(x) + {b:.6g}$"
    return dict(nombre="Logarítmico", params=(a, b), r2=r2, eq=eq,
                func=lambda xv, a=a, b=b: a * np.log(xv) + b)


MODELOS = [ajustar_lineal, ajustar_potencial, ajustar_exponencial,
           ajustar_logaritmico]


def mejor_ajuste(x, y):
    resultados = []
    for modelo in MODELOS:
        try:
            r = modelo(x, y)
            if r is not None and np.isfinite(r["r2"]):
                resultados.append(r)
        except Exception:
            continue
    resultados.sort(key=lambda d: d["r2"], reverse=True)
    return resultados


# ------------------------------------------------------------------
# 3. FUNCIÓN DE SUBPANEL
# ------------------------------------------------------------------

def poblar_subpanel(ax, x, y, xlabel, ylabel, titulo, fill_regions=False, legend_loc="upper right"):
    resultados = mejor_ajuste(x, y)
    if not resultados:
        ax.set_title(rf"{titulo}\n[sin ajuste]", fontsize=10)
        return None

    mejor = resultados[0]
    x_fino = np.linspace(np.min(x), np.max(x), PLOT_POINTS)
    y_fino = mejor["func"](x_fino)

    ax.scatter(x, y, color="#1f4e79", s=35, zorder=3)

    if fill_regions:
        margen_y = 0.08 * (np.max(y) - np.min(y)) if np.ptp(y) > 0 else 0.1
        ymin = np.min([np.min(y), np.min(y_fino)]) - margen_y
        ymax = np.max([np.max(y), np.max(y_fino)]) + margen_y
        ax.set_ylim(ymin, ymax)
        ax.fill_between(x_fino, y_fino, ymax, color="#2f6fed", alpha=0.18,
                        zorder=1)
        ax.fill_between(x_fino, ymin, y_fino, color="#d9534f", alpha=0.18,
                        zorder=1)

    ax.plot(x_fino, y_fino, color="#c0392b", linewidth=1.8,
            zorder=2)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(rf"{titulo}", fontsize=25)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(np.min(x), np.max(x))

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f4e79",
               markeredgecolor="#1f4e79", markersize=8, label="Datos"),
        Line2D([0], [0], color="#c0392b", linewidth=1.8, label=rf"Ajuste {mejor['nombre']}"),
    ]
    if fill_regions:
        legend_handles.extend([
            Patch(facecolor="#2f6fed", edgecolor="none", alpha=0.18,
                  label="Interacción"),
            Patch(facecolor="#d9534f", edgecolor="none", alpha=0.18,
                  label="Sin interacción"),
        ])

    legend_handles.extend([
        Line2D([0], [0], linestyle="None", marker="", color="none",
               label=mejor["eq"]),
        Line2D([0], [0], linestyle="None", marker="", color="none",
               label=rf"$R^2$ = {mejor['r2']:.4f}"),
    ])

    ax.legend(handles=legend_handles, fontsize=11, loc=legend_loc,
              ncol=1, borderaxespad=0.4, handlelength=1.5,
              frameon=True, framealpha=0.95)

    print(f"  -> {titulo} | {mejor['nombre']} | R^2={mejor['r2']:.4f}")
    return dict(relacion=titulo, modelo=mejor["nombre"],
                ecuacion=mejor["eq"], r2=mejor["r2"], todos=resultados)


# ------------------------------------------------------------------
# 4. EJECUCIÓN PRINCIPAL
# ------------------------------------------------------------------

variables_y = [
    (Vmax95,             r"$V_{\mathrm{max}}^{95\%}$ [km/s]",      "Vmax95"),
    (T_interaccion,      r"Tiempo de interacción [h]",             "T_interaccion"),
    (Altura_interaccion, r"Altura de interacción [AU]",            "Altura_interaccion"),
    (T95_Vmin,           r"Tiempo $V_{\mathrm{min}}^{95\%}$ [h]",   "T95_Vmin"),
]


def main():
    resumen = []

    # ── Figura 1: a_d vs cada variable (2×2) ─────────────────────
    print("\n=== Figura 1: relaciones con a_d ===")
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
    axes1 = axes1.flatten()
    for i, (y_data, y_label, y_tag) in enumerate(variables_y):
        titulo = f"$a_d$ vs {y_label}"
        r = poblar_subpanel(axes1[i], a_d, y_data,
                            r"$a_d$ [km/s$^2$]", y_label, titulo)
        if r:
            r["relacion"] = f"ad_vs_{y_tag}"
            resumen.append(r)
    fig1.suptitle(r"Relaciones con el parámetro de decaimiento $a_d$",
                  fontsize=25)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "fig_ad_relaciones.png"),
                 dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig1)

    # ── Figura 2: t_d vs cada variable (2×2) ─────────────────────
    print("\n=== Figura 2: relaciones con t_d ===")
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2 = axes2.flatten()
    for i, (y_data, y_label, y_tag) in enumerate(variables_y):
        titulo = f"$\\tau_d$ vs {y_label}"
        r = poblar_subpanel(axes2[i], t_d, y_data,
                            r"$\tau_d$ [$10^3$ s]", y_label, titulo, legend_loc="lower right")
        axes2[i].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000:.1f}"))
        if r:
            r["relacion"] = f"td_vs_{y_tag}"
            resumen.append(r)
    fig2.suptitle(r"Relaciones con el parámetro de decaimiento $\tau_d$",
                  fontsize=25)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "fig_td_relaciones.png"),
                 dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig2)

    # ── Figura 3: a_d vs t_d (individual) ────────────────────────
    print("\n=== Figura 3: a_d vs t_d ===")
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    r = poblar_subpanel(ax3, a_d, t_d,
                        r"$a_d$ [km/s$^2$]", r"$\tau_d$ [s]", r"$a_d$ vs $\tau_d$",
                        fill_regions=True)
    if r:
        r["relacion"] = "ad_vs_td"
        resumen.append(r)

    ax3.set_ylabel(r"$\tau_d$ [$10^3$ s]")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000:.1f}"))

    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "fig_ad_vs_td.png"),
                 dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig3)

    # ── Resumen TXT ───────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "resumen_ajustes.txt"), "w",
              encoding="utf-8") as f:
        f.write("RESUMEN DE AJUSTES — Interacción CME-CME\n")
        f.write("=" * 55 + "\n\n")
        for r in resumen:
            f.write(f"Relación: {r['relacion']}\n")
            f.write(f"  Mejor modelo : {r['modelo']}\n")
            f.write(f"  Ecuación     : {r['ecuacion']}\n")
            f.write(f"  R^2          : {r['r2']:.4f}\n")
            f.write("  Comparación:\n")
            for m in r["todos"]:
                f.write(f"    - {m['nombre']:<12s} R^2={m['r2']:.4f}   {m['eq']}\n")
            f.write("\n")

    # ── Resumen CSV ───────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "resumen_ajustes.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Relacion", "Mejor_modelo", "Ecuacion", "R2"])
        for r in resumen:
            writer.writerow([r["relacion"], r["modelo"],
                             r["ecuacion"], f"{r['r2']:.4f}"])

    print(f"\nListo. Archivos en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()