import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

# Configuración LaTeX global
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# --- Lectura SSN ---
df = pd.read_csv('SN_y_tot_V2.0.csv', sep=';')
min_x = int(df['Año'].min())
max_x = int(df['Año'].max())

# --- Lectura CMEs ---
df_cme = pd.read_csv('eventos_por_año_Univariado.csv', sep=',')

if df_cme['Año'].dtype == object:
    df_cme['Año'] = pd.to_datetime(df_cme['Año'], errors='coerce').dt.year
else:
    try:
        df_cme['Año'] = pd.to_datetime(df_cme['Año']).dt.year
    except Exception:
        df_cme['Año'] = df_cme['Año'].astype(int)

df_cme = df_cme.dropna(subset=['Año'])
df_cme['Año'] = df_cme['Año'].astype(int)

if 'cantidad_eventos' in df_cme.columns:
    cme_col = 'cantidad_eventos'
else:
    num_cols = df_cme.select_dtypes(include=['number']).columns.tolist()
    cme_col = num_cols[0] if num_cols else None

# --- Rango común (zoom automático) ---
year_min = int(max(df['Año'].min(), df_cme['Año'].min()))
year_max = int(min(df['Año'].max(), df_cme['Año'].max()))

# ================================================================
# FIGURA: 2 paneles verticales
# ================================================================
fig, (ax_main, ax_sub) = plt.subplots(
    2, 1, figsize=(11, 8),
    gridspec_kw={'height_ratios': [2, 1]}
)

# ── Panel principal ──────────────────────────────────────────────
df.plot(x='Año', y=' SSN_Y', ax=ax_main, legend=False,
        color='C0', label=r'Sunspots anuales')
ax_main.set_xlabel(r'A\~no', fontsize=25, color='black')
ax_main.set_ylabel(r'Sunspots por a\~no', fontsize=25, color='black')
ax_main.set_title(r'Sunspots y CMEs anuales', fontsize=26, color='black')
ax_main.tick_params(axis='both', direction='in', which='both', labelsize=14, colors='black')
ax_main.yaxis.label.set_color('black')
ax_main.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
ax_main.set_xticks(np.arange(min_x, max_x + 1, 22))
ax_main.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

# ── Región de zoom más visible ───────────────────────────────────
ax_main.axvspan(year_min, year_max, color='steelblue', alpha=0.15, label=r'Región de acercamiento')

ax2 = ax_main.twinx()
if cme_col:
    ax2.plot(df_cme['Año'], df_cme[cme_col], color='C1', label=r'CMEs anuales')
    ax2.set_ylabel(r'CMEs por a\~no', fontsize=25, color='black')
    ax2.tick_params(axis='y', labelsize=14, colors='black')
    ax2.yaxis.label.set_color('black')

lines_1, labels_1 = ax_main.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
if lines_1 or lines_2:
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2,
               loc='upper left', fontsize=15)

# ── Subpanel: zoom ───────────────────────────────────────────────
df_zoom     = df[(df['Año'] >= year_min) & (df['Año'] <= year_max)]
df_cme_zoom = df_cme[(df_cme['Año'] >= year_min) & (df_cme['Año'] <= year_max)]

ax_sub.plot(df_zoom['Año'], df_zoom[' SSN_Y'],
            color='C0', lw=2, label=r'Sunspots anuales')
ax_sub.set_ylabel(r'Sunspots por a\~no', fontsize=23, color='black')
ax_sub.yaxis.label.set_color('black')
ax_sub.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
ax_sub.set_xlim(year_min, year_max)
ax_sub.set_xticks(np.arange(year_min, year_max + 1, 2))
ax_sub.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

# ── Ticks principales y subticks con tamaños diferenciados ───────
ax_sub.xaxis.set_minor_locator(AutoMinorLocator(4))
ax_sub.tick_params(axis='x', which='major', direction='in', length=6, labelsize=13, colors='black')
ax_sub.tick_params(axis='x', which='minor', direction='in', length=3, colors='black')
ax_sub.tick_params(axis='y', labelcolor='black', labelsize=13, colors='black')

ax_sub.set_xlabel(r'A\~no', fontsize=25, color='black')
ax_sub.set_title(
    rf'Acercamiento: {year_min}--{year_max}',
    fontsize=20, color='black'
)

ax_sub2 = ax_sub.twinx()
ax_sub2.plot(df_cme_zoom['Año'], df_cme_zoom[cme_col],
             color='C1', lw=2, label=r'CMEs anuales')
ax_sub2.set_ylabel(r'CMEs por a\~no', fontsize=23, color='black')
ax_sub2.tick_params(axis='y', labelcolor='black', labelsize=13, colors='black')
ax_sub2.yaxis.label.set_color('black')

plt.tight_layout()
plt.savefig('SSN_y_CMEs.png', dpi=300)
plt.show()