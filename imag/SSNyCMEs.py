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

# --- Primera gráfica ---
fig, ax = plt.subplots(figsize=(12, 5))
df.plot(x='Año', y=' SSN_Y', ax=ax, legend=False)
ax.set_xlabel(r'A\~no', fontsize=18)
ax.set_ylabel(r'N\'umero de manchas solares por a\~no', fontsize=18)
ax.set_title(r'Perfil anual de manchas solares', fontsize=19)
ax.tick_params(axis='both', direction='in', which='both', labelsize=14)
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
min_x = int(df['Año'].min())
max_x = int(df['Año'].max())
ax.set_xticks(np.arange(min_x, max_x+1, 22))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
plt.tight_layout()
plt.savefig('SSN.png', dpi=300)
plt.show()

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

# --- Segunda gráfica ---
fig, ax = plt.subplots(figsize=(12, 5))
df.plot(x='Año', y=' SSN_Y', ax=ax, legend=False, color='C0', label=r'Manchas solares anuales')
ax.set_xlabel(r'A\~no', fontsize=20)
ax.set_ylabel(r'N\'umero de manchas solares por a\~no', fontsize=20)
ax.set_title(r'Manchas solares y CMEs anuales', fontsize=21)
ax.tick_params(axis='y', labelsize=14)

ax2 = ax.twinx()
if cme_col:
    ax2.plot(df_cme['Año'], df_cme[cme_col], color='C1', label=r'CMEs anuales')
    ax2.set_ylabel(r'Cantidad de CMEs por a\~no', fontsize=20)
    ax2.tick_params(axis='y', labelsize=14)
else:
    ax2.set_ylabel(r'CMEs (datos no encontrados)', fontsize=20)

ax.tick_params(axis='both', direction='in', which='both', labelsize=14)
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
ax.set_xticks(np.arange(min_x, max_x+1, 22))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
if lines_1 or lines_2:
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=13)

plt.tight_layout()
plt.savefig('SSN_y_CMEs.png', dpi=300)
plt.show()