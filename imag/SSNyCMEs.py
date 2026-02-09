import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('SN_y_tot_V2.0.csv', sep=';')
fig, ax = plt.subplots(figsize=(12, 5))  # Más ancho
df.plot(x='Año', y=' SSN_Y', ax=ax, legend=False)
ax.set_xlabel('Año', fontweight='bold', fontsize=15)
ax.set_ylabel('Número de manchas solares por año', fontweight='bold', fontsize=15)
ax.tick_params(axis='both', direction='in', which='both')  # ticks internos
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # Cuadrícula
# Más ticks en x
import numpy as np
from matplotlib.ticker import FuncFormatter
min_x = int(df['Año'].min())
max_x = int(df['Año'].max())
ax.set_xticks(np.arange(min_x, max_x+1, 22))  # cada 5 años
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))  # solo enteros
plt.tight_layout()
plt.savefig('SSN.png', dpi=300)
plt.show()
df_cme = pd.read_csv('eventos_por_año_Univariado.csv', sep=',')

# Asegurarse de tener columna de año como entero
if df_cme['Año'].dtype == object:
	df_cme['Año'] = pd.to_datetime(df_cme['Año'], errors='coerce').dt.year
else:
	try:
		df_cme['Año'] = pd.to_datetime(df_cme['Año']).dt.year
	except Exception:
		df_cme['Año'] = df_cme['Año'].astype(int)

df_cme = df_cme.dropna(subset=['Año'])
df_cme['Año'] = df_cme['Año'].astype(int)

# Renombrar columna de cantidad si hace falta
if 'cantidad_eventos' in df_cme.columns:
	cme_col = 'cantidad_eventos'
else:
	# intentar detectar la primera columna numérica diferente a Año
	num_cols = df_cme.select_dtypes(include=['number']).columns.tolist()
	cme_col = num_cols[0] if num_cols else None

fig, ax = plt.subplots(figsize=(12, 5))  # Más ancho


# Graficar manchas solares (serie original)
df.plot(x='Año', y=' SSN_Y', ax=ax, legend=False, color='C0', label='Manchas solares anuales')
ax.set_xlabel('Año', fontweight='bold', fontsize=15)
ax.set_ylabel('Número de manchas solares por año', fontweight='bold', fontsize=15)
ax.tick_params(axis='y')

# Eje secundario para CMEs
ax2 = ax.twinx()
if cme_col:
	# asegurarse de alinear por año
	ax2.plot(df_cme['Año'], df_cme[cme_col], color='C1', label='CMEs anuales')
	ax2.set_ylabel('Cantidad de CMEs por año', fontweight='bold', fontsize=15)
	ax2.tick_params(axis='y')
else:
	ax2.set_ylabel('CMEs (datos no encontrados)', fontweight='bold', fontsize=15)

# Estética: ticks, cuadrícula y formato de eje x
ax.tick_params(axis='both', direction='in', which='both')  # ticks internos
ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)  # Cuadrícula
min_x = int(df['Año'].min())
max_x = int(df['Año'].max())
# Separar más los ticks del eje x (cada 10 años)
ax.set_xticks(np.arange(min_x, max_x+1, 22))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

# Leyenda combinada
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
if lines_1 or lines_2:
	ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.savefig('SSN_y_CMEs.png', dpi=300)
plt.show()