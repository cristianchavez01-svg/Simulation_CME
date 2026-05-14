import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Configuración LaTeX global
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

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

fig, ax = plt.subplots(figsize=(11, 5.5))
df.plot(x='Año', y=' SSN_Y', ax=ax, legend=False, color='C0', label=r'Sunspots anuales')
ax.set_xlabel(r'A\~no', fontsize=23)
ax.set_ylabel(r'Sunspots por a\~no', fontsize=23)
ax.set_title(r'Sunspots y CMEs anuales', fontsize=23)
ax.tick_params(axis='y', labelsize=16)

ax2 = ax.twinx()
if cme_col:
    ax2.plot(df_cme['Año'], df_cme[cme_col], color='C1', label=r'CMEs anuales')
    ax2.set_ylabel(r'CMEs por a\~no', fontsize=23)
    ax2.tick_params(axis='y', labelsize=16)
else:
    ax2.set_ylabel(r'CMEs (datos no encontrados)', fontsize=23)

ax.tick_params(axis='both', direction='in', which='both', labelsize=16)
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
ax.set_xticks(np.arange(min_x, max_x+1, 22))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
if lines_1 or lines_2:
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=21)

plt.tight_layout()
plt.savefig('SSN_y_CMEs.png', dpi=300)
plt.show()