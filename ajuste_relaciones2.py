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
a_d = np.array([0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017,
                0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024, 0.025,
                0.026, 0.027, 0.028, 0.029, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

t_d = np.array([77521, 70156, 64300, 59700, 55800, 52580, 49820, 47420,
                 45360, 43510, 41876, 40415, 39085, 37885, 36795, 35785,
                 34868, 34012, 33222, 32489, 17251, 13287, 11647, 10689, 10039, 9559.4, 9185.37, 8882.42])

Vmax95 = np.array([731.92, 723.49, 710.72, 701.71, 693.06, 687.09, 681.29,
                    676.15, 672.17, 668.38, 664.46, 662.29, 659.43, 657.01,
                    654.99, 653.02, 651.50, 649.78, 648.07, 646.94, 620.87, 614.45, 612.07, 611.31, 609.82, 609.09, 608.62, 608.68])

T_interaccion = np.array([89.70, 89.59, 89.91, 89.55, 89.90, 89.81, 89.82,
                           90.00, 89.69, 89.83, 89.79, 89.67, 89.80, 89.78,
                           89.66, 89.84, 89.68, 89.79, 89.79, 89.75, 89.81, 89.74, 89.60, 89.76, 90.65, 90.06, 90.01, 87.58])

Altura_interaccion = np.array([1.180, 1.179, 1.183, 1.178, 1.183, 1.182, 1.182,
                                1.185, 1.180, 1.182, 1.182, 1.180, 1.182, 1.182,
                                1.180, 1.182, 1.180, 1.182, 1.182, 1.181, 1.182, 1.181, 1.179, 1.181, 1.195, 1.186, 1.185, 1.148])

T95_Vmin = np.array([64.66, 63.26, 59.60, 56.59, 53.91, 51.89, 49.87, 48.10,
                      46.56, 45.22, 43.94, 42.91, 41.87, 40.96, 40.15, 39.43,
                      38.79, 38.15, 37.51, 37.02, 26.07, 23.28, 22.17, 21.58, 21.09, 20.77, 20.53, 20.36])

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
    fig1.savefig(os.path.join(OUTPUT_DIR, "fig_ad_relaciones2.png"),
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
    fig2.savefig(os.path.join(OUTPUT_DIR, "fig_td_relaciones2.png"),
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
    fig3.savefig(os.path.join(OUTPUT_DIR, "fig_ad_vs_td2.png"),
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