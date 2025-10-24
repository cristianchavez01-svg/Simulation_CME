import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.integrate import quad
from matplotlib.patches import Circle

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
tr1 = 0.1
td1 = 5
ar1 = 0.1
ad1 = 1
v01 = 0.2
tr2 = 0.2
td2 = 7
ar2 = 0.1
ad2 = 3
v02 = 0

# Tiempos específicos para las 4 imágenes (ajustados para mejor visualización)
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

# Configuración de estilo consistente para todas las imágenes
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Generar las 4 imágenes con estilo consistente
for i, t in enumerate(tiempos):
    # Crear figura con tamaño optimizado para grid en LaTeX
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
    
    # Configurar el gráfico polar (MISMO para todas las imágenes)
    ax.set_rmax(30000)  # Mismo rango para todas
    ax.set_thetamin(75)
    ax.set_thetamax(-75)
    ax.grid(True, alpha=0.3)
    
    # Añadir el círculo del sol
    circle = Circle((0, 0), 0.1, transform=ax.transData._b, color="red", alpha=1)
    ax.add_patch(circle)
    
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
    
    # GRAFICAR PUNTOS INDIVIDUALES en lugar de curvas
    ax.scatter(theta1_new, r11, s=1, color="blue", alpha=0.6, label="Puntos 1")
    ax.scatter(theta2_new, r22, s=1, color="red", alpha=0.6, label="Puntos 2")
    
    # Título consistente
    ax.set_title(f"t = {t:.1f} s", pad=20)
    
    # Leyenda solo en la primera imagen para evitar repetición
    if i == 0:
        ax.legend(loc='upper right', fontsize=8, markerscale=3)  # markerscale para hacer los puntos más visibles en la leyenda
    
    # Guardar la imagen con nombre consistente
    filename = f"puntos_t{i+1}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Imagen guardada: {filename}")
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)

print("\nTodas las imágenes han sido generadas exitosamente!")

# Generar también una figura con las 4 subfiguras (opcional)
print("\nGenerando figura combinada...")
fig, axs = plt.subplots(2, 2, figsize=(10, 8), 
                       subplot_kw=dict(polar=True))
axs = axs.flatten()

for i, (ax, t) in enumerate(zip(axs, tiempos)):
    # Configurar cada subplot
    ax.set_rmax(30000)
    ax.set_thetamin(75)
    ax.set_thetamax(-75)
    ax.grid(True, alpha=0.3)
    
    # Círculo del sol
    circle = Circle((0, 0), 0.1, transform=ax.transData._b, color="red", alpha=1)
    ax.add_patch(circle)
    
    # Cálculos (igual que antes)
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
    
    # GRAFICAR PUNTOS en lugar de curvas
    ax.scatter(theta1_new, r11, s=0.1, color="blue", alpha=0.6, label="Puntos 1")
    ax.scatter(theta2_new, r22, s=0.1, color="red", alpha=0.6, label="Puntos 2")
    ax.set_title(f"t = {t:.1f} s", fontsize=11)
    
    # Leyenda solo en el primer subplot
    if i == 0:
        ax.legend(loc='upper right', fontsize=8, markerscale=3)

plt.tight_layout()
plt.savefig('puntos_combinada.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figura combinada guardada: puntos_combinada.png")