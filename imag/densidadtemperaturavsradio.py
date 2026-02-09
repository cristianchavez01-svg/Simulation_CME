import numpy as np
import matplotlib.pyplot as plt

def leer_datos_solar(archivo):
    """Lee el archivo del modelo solar, ignorando encabezados y texto."""
    datos = []
    with open(archivo, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            
            # Saltar líneas que contienen texto común en encabezados
            palabras_texto = ['Mass', 'Radius', 'Temperature', 'Density', 
                             'Pressure', 'Luminosity', 'M/Msun', 'R/Rsun',
                             'Standard', 'Solar', 'Model', 'Columns']
            
            if any(palabra in linea for palabra in palabras_texto):
                continue
            
            # Intentar convertir a números
            partes = linea.split()
            if len(partes) < 4:
                continue
            
            fila = []
            es_valida = True
            for i in range(4):  # Solo necesitamos 4 columnas
                try:
                    valor = float(partes[i])
                    fila.append(valor)
                except ValueError:
                    es_valida = False
                    break
            
            if es_valida and len(fila) == 4:
                datos.append(fila)
    
    return np.array(datos)

# Leer datos
print("Leyendo archivo 'datosTrhovsR.dat'...")
datos = leer_datos_solar('imag/datosTrhovsR.dat')

if len(datos) == 0:
    print("ERROR: No se pudieron leer datos. Verifica el formato del archivo.")
    print("El archivo debe tener al menos 4 columnas numéricas por fila.")
    exit()

print(f"✓ {len(datos)} filas de datos cargadas")

# Extraer variables
R = datos[:, 1]    # Columna 2: Radio (R/Rsun)
T = datos[:, 2]    # Columna 3: Temperatura (K)
rho = datos[:, 3]  # Columna 4: Densidad (g/cm³)

# Crear gráfica
fig, ax1 = plt.subplots(figsize=(11, 7))

# Graficar densidad (azul) - con etiqueta para la leyenda
linea_densidad, = ax1.plot(R, rho, 'b-', linewidth=2, label='Densidad')
ax1.axvline(x=0.71, color='k', linestyle='-.', linewidth=1)
ax1.set_xlabel('Radio (R/R$_\odot$)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Densidad (g cm$^{-3}$)', fontsize=13, fontweight='bold')
ax1.tick_params(axis='y')
ax1.set_yscale('log')
ax1.set_xlim(left=0)
ax1.tick_params(axis='both', direction='in', length=8, width=1.2, labelsize=14)  # Ticks mayores
ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1)  # Ticks menores
ax1.grid(True, alpha=0.3)

# Graficar temperatura (rojo) - con etiqueta para la leyenda
ax2 = ax1.twinx()
linea_temperatura, = ax2.plot(R, T, 'r-', linewidth=2, label='Temperatura')
ax2.set_ylabel('Temperatura (K)', fontsize=13, fontweight='bold')
ax2.tick_params(axis='y')
ax2.set_yscale('log')
ax2.set_xlim(ax1.get_xlim())
ax2.tick_params(axis='both', direction='in', length=8, width=1.2, labelsize=14)  # Ticks mayores
ax2.tick_params(axis='both', which='minor', direction='in', length=4, width=1)  # Ticks menores
ax2.grid(False)

# Crear una leyenda unificada para ambas líneas
# Primero, creamos handles y labels para ambas líneas
handles = [linea_densidad, linea_temperatura]
labels = ['Densidad', 'Temperatura']

# Agregar la leyenda en una posición adecuada
ax1.legend(handles, labels, 
           loc='upper right',  # Posición de la leyenda
           fontsize=12,
           frameon=True,
           fancybox=False,
           shadow=False,
           framealpha=0.95,
           edgecolor='black')

# Otra opción para la leyenda (más compacta):
# ax1.legend(handles, labels, loc='best', fontsize=11)

plt.tight_layout()

# Guardar
plt.savefig('Densidad_temperatura_vs_radio_Modelo_solar.png', dpi=300, bbox_inches='tight')
print("✓ Gráfica guardada como 'Densidad_temperatura_vs_radio_Modelo_solar.png'")

plt.show()