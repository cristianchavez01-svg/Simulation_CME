import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.integrate import quad
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# Configuración inicial
theta1 = []
theta2 = []
r1 = []  # Radio 1 en función de θ
r2 = []  # Radio 2 en función de θ

# Generar puntos aleatorios
for i in range(10000):
    r1.append(rd.uniform(0, 9))
    theta1.append(rd.uniform(0, 2 * np.pi))
    r2.append(rd.uniform(0, 11))
    theta2.append(rd.uniform(0, 2 * np.pi))

theta1ord = np.append(sorted(theta1), sorted(theta1)[0])
r1_orig = np.append(r1, r1[0])
theta2ord = np.append(sorted(theta2), sorted(theta2)[0])
r2_orig = np.append(r2, r2[0])

# Parámetros para la aceleración
tr1 = 0.2
td1 = 3
ar1 = 1
ad1 = 1
v01 = 0

tr2 = 0.2
td2 = 5
ar2 = 0.4
ad2 = 4
v02 = 0

# Tiempos específicos para las 4 imágenes
tiempos = [3.0, 6.0, 9.0, 17.0]

# Funciones de aceleración y velocidad
def f(s):
    return (ar1 * ad1) / (ad1 * np.exp(-s / tr1) + ar1 * np.exp(s / td1))

def x_of1(t):
    integrand = lambda s: (t - s) * f(s)
    val, err = quad(integrand, 0, t)
    return val

def g(s):
    return (ar2 * ad2) / (ad2 * np.exp(-s / tr2) + ar2 * np.exp(s / td2))

def x_of2(t):
    integrand = lambda s: (t - s) * g(s)
    val, err = quad(integrand, 0, t)
    return val

def v1(s):
    return v01 + quad(f, 0, s)[0]

def v2(s):
    return v02 + quad(g, 0, s)[0]

def expansion_factor1(time):
    return time**3

def expansion_factor2(time):
    return time**3

# =============================================================================
# PARTE 1: GENERAR LAS 4 IMÁGENES INDIVIDUALES
# =============================================================================

# Configuración de estilo consistente para todas las imágenes
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Generar las 4 imágenes con estilo consistente
for i, t in enumerate(tiempos):
    
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
    
    # Configurar el gráfico polar (MISMO para todas las imágenes)
    ax.set_rmax(30000)  # Mismo rango para todas
    ax.set_thetamin(75)
    ax.set_thetamax(-75)
    ax.grid(True, alpha=0.3)

    # Calcular desplazamientos
    dx1 = v01 * t + x_of1(t)
    dx2 = v02 * t + x_of2(t)
    
    # Calcular nuevas coordenadas
    x1 = r1_orig * np.cos(theta1ord) + dx1
    y1 = r1_orig * np.sin(theta1ord)
    x2 = r2_orig * np.cos(theta2ord) + dx2
    y2 = r2_orig * np.sin(theta2ord)
    
    # Convertir a polares
    r1_new = np.sqrt(x1**2 + y1**2)
    theta1_new = np.arctan2(y1, x1)
    r2_new = np.sqrt(x2**2 + y2**2)
    theta2_new = np.arctan2(y2, x2)
    
    # Aplicar expansión radial
    r11 = r1_new * t + expansion_factor1(t)
    r22 = r2_new * t + expansion_factor2(t)
    
    # GRAFICAR PUNTOS INDIVIDUALES
    ax.scatter(theta1_new, r11, s=0.1, color="blue", alpha=0.6, label="CME1")
    ax.scatter(theta2_new, r22, s=0.1, color="red", alpha=0.6, label="CME2")
    
    ax.set_title(f"t = {t:.1f} s", pad=20)
    
    if i == 0:
        ax.legend(loc='upper right', fontsize=8, markerscale=3)  

    # Guardar la imagen con nombre consistente
    filename = f"puntos_t{i+1}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)

# =============================================================================
# PARTE 2: GENERAR EL GIF ANIMADO
# =============================================================================

fps = 20
tiempo_inicial = 0

# Listas para almacenar datos
f_values = []
g_values = []
time_values = []
v_values_1 = []
v_values_2 = []

# Crear figura para la animación
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
ax.set_rmax(30000)
ax.set_thetamin(75)
ax.set_thetamax(-75)
ax.grid(True, alpha=0.3)


# Usar scatter para los puntos
scat1 = ax.scatter([], [], s=0.1, color="blue", alpha=0.4, label="Puntos 1")
scat2 = ax.scatter([], [], s=0.1, color="red", alpha=0.4, label="Puntos 2")
ax.legend(loc='upper right', fontsize=8, markerscale=3)

def init():
    scat1.set_offsets(np.empty((0, 2)))
    scat2.set_offsets(np.empty((0, 2)))
    return scat1, scat2

def update(frame):
    # tiempo en segundos
    t = (frame / fps) + tiempo_inicial
    time_values.append(t)

    # Calcular desplazamientos
    dx1 = v01 * t + x_of1(t)
    dx2 = v02 * t + x_of2(t)

    # Calcular nuevas coordenadas
    x1 = r1_orig * np.cos(theta1ord) + dx1
    y1 = r1_orig * np.sin(theta1ord)
    x2 = r2_orig * np.cos(theta2ord) + dx2
    y2 = r2_orig * np.sin(theta2ord)
    
    # Convertir a polares
    r1_new = np.sqrt(x1**2 + y1**2)
    theta1_new = np.arctan2(y1, x1)
    r2_new = np.sqrt(x2**2 + y2**2)
    theta2_new = np.arctan2(y2, x2)
    
    # Aplicar expansión radial
    r11 = r1_new * t + expansion_factor1(t)
    r22 = r2_new * t + expansion_factor2(t)
    
    # Actualizar los puntos scatter
    points1 = np.column_stack([theta1_new, r11])
    points2 = np.column_stack([theta2_new, r22])
    
    scat1.set_offsets(points1)
    scat2.set_offsets(points2)
    
    ax.set_title(f"t = {t:.1f} s", pad=20)

    # Almacenar los valores de f(t) y g(t)
    f_values.append(f(t))
    g_values.append(g(t))

    # Almacenar los valores de velocidad
    v_values_1.append(v1(t))
    v_values_2.append(v2(t))
    
    return scat1, scat2

# Crear animación
ani = FuncAnimation(fig, update, frames=600, init_func=init, blit=False, interval=50)

# Guardar GIF
out_path = "puntos_polar.gif"
ani.save(out_path, writer=PillowWriter(fps=fps), dpi=150)
plt.close(fig)

# =============================================================================
# PARTE 3: GENERAR GRÁFICOS DE ACELERACIÓN Y VELOCIDAD, con máximos
# =============================================================================

# Convertir a arrays numpy para cálculos
time_array = np.array(time_values)
f_array = np.array(f_values)
g_array = np.array(g_values)
v1_array = np.array(v_values_1)
v2_array = np.array(v_values_2)

# Calcular máximos de ACELERACIÓN
max_f_idx = np.argmax(f_array)
max_g_idx = np.argmax(g_array)

max_f_time = time_array[max_f_idx]
max_f_value = f_array[max_f_idx]
max_g_time = time_array[max_g_idx]
max_g_value = g_array[max_g_idx]

# Calcular máximos de VELOCIDAD
max_v1_idx = np.argmax(v1_array)
max_v2_idx = np.argmax(v2_array)

max_v1_time = time_array[max_v1_idx]
max_v1_value = v1_array[max_v1_idx]
max_v2_time = time_array[max_v2_idx]
max_v2_value = v2_array[max_v2_idx]

# Gráfico de aceleración
plt.figure(figsize=(8, 4))
plt.plot(time_array, f_array, label='a1(t)', color='blue', linewidth=2)
plt.plot(time_array, g_array, label='a2(t)', color='red', linewidth=2)
# Marcar máximos
plt.scatter(max_f_time, max_f_value, color='blue', s=100, zorder=5, 
           label=f'Máx a1: {max_f_value:.3f} en t={max_f_time:.1f}s')
plt.scatter(max_g_time, max_g_value, color='red', s=100, zorder=5,
           label=f'Máx a2: {max_g_value:.3f} en t={max_g_time:.1f}s')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración')
plt.title('Aceleración en función del tiempo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('aceleracion_vs_tiempo.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico de velocidad
plt.figure(figsize=(8, 4))
plt.plot(time_array, v1_array, label='v1(t)', color='blue', linewidth=2)
plt.plot(time_array, v2_array, label='v2(t)', color='red', linewidth=2)
# Marcar máximos
plt.scatter(max_v1_time, max_v1_value, color='blue', s=100, zorder=5,
           label=f'Máx v1: {max_v1_value:.3f} en t={max_v1_time:.1f}s')
plt.scatter(max_v2_time, max_v2_value, color='red', s=100, zorder=5,
           label=f'Máx v2: {max_v2_value:.3f} en t={max_v2_time:.1f}s')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad')
plt.title('Velocidad en función del tiempo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('velocidad_vs_tiempo.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PARTE 4: GENERAR FIGURA COMBINADA CON LAS 4 IMÁGENES
# =============================================================================

fig, axs = plt.subplots(2, 2, figsize=(10, 8), subplot_kw=dict(polar=True))
axs = axs.flatten()

for i, (ax, t) in enumerate(zip(axs, tiempos)):
    # Configurar cada subplot
    ax.set_rmax(30000)
    ax.set_thetamin(75)
    ax.set_thetamax(-75)
    ax.grid(True, alpha=0.3)
    
    # Cálculos
    dx1 = v01 * t + x_of1(t)
    dx2 = v02 * t + x_of2(t)
    
    x1 = r1_orig * np.cos(theta1ord) + dx1
    y1 = r1_orig * np.sin(theta1ord)
    x2 = r2_orig * np.cos(theta2ord) + dx2
    y2 = r2_orig * np.sin(theta2ord)
    
    r1_new = np.sqrt(x1**2 + y1**2)
    theta1_new = np.arctan2(y1, x1)
    r2_new = np.sqrt(x2**2 + y2**2)
    theta2_new = np.arctan2(y2, x2)
    
    r11 = r1_new * t + expansion_factor1(t)
    r22 = r2_new * t + expansion_factor2(t)
    
    # Graficar puntos
    ax.scatter(theta1_new, r11, s=0.1, color="blue", alpha=0.6, label="Puntos 1")
    ax.scatter(theta2_new, r22, s=0.1, color="red", alpha=0.6, label="Puntos 2")
    ax.set_title(f"t = {t:.1f} s", fontsize=11)
    
    # Leyenda solo en el primer subplot
    if i == 0:
        ax.legend(loc='upper right', fontsize=8, markerscale=3)

plt.tight_layout()
plt.savefig('puntos_combinada.png', dpi=300, bbox_inches='tight')
plt.close()
