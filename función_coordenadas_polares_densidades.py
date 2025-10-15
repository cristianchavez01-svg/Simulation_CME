import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.integrate import quad
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

fps = 15
theta1 = []
theta2 = []
r1 = []  # Radio 1 en función de θ
r2 = []  # Radio 2 en función de θ

for i in range(100):
    r1.append(rd.uniform(5, 5.05))
    theta1.append(rd.uniform(0, 2 * np.pi))
    r2.append(rd.uniform(6, 6.055))
    theta2.append(rd.uniform(0, 2 * np.pi))

theta1ord = np.append(sorted(theta1), sorted(theta1)[0])  # Ordenar los ángulos
r1 = np.append(r1, r1[0])
theta2ord = np.append(sorted(theta2), sorted(theta2)[0])
r2 = np.append(r2, r2[0])

fig = plt.figure()
ax = plt.subplot(111, polar=True)  # Crear un gráfico polar
ax.set_rmax(50)  # radio maximo del plano polar
ax.set_thetamin(90)  # límite inferior en grados, para que se muestre solamente la mitad del plano.
ax.set_thetamax(-90)  # límite superior en grados
circle = Circle((0, 0), 0.1, transform=ax.transData._b, color="red", alpha=1)
ax.add_patch(circle)

line1, = ax.plot([], [], lw=2, color="blue", label="Curva 1")
line2, = ax.plot([], [], lw=2, color="red", label="Curva 2")

# Inicializar la imagen para el mapa de densidad
num_bins = 50
heatmap_inicial = np.zeros((num_bins, num_bins))
img = ax.imshow(heatmap_inicial, extent=[-50, 50, -50, 50], origin='lower', cmap=cm.viridis, aspect='auto')

# Ajustar los límites del gráfico para que coincidan con el extent
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])


# para que las curvas se desplacen:
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    img.set_data(heatmap_inicial)
    return line1, line2, img


## parametros para la aceleración

# Para la CME1
tr1 = 0.1
td1 = 5
ar1 = 0.1
ad1 = 1
v01 = 0.2
# Para la CME2
tr2 = 0.2
td2 = 7
ar2 = 0.1
ad2 = 3
v02 = 0


tiempo_inicial = 0  # tiempo inicial en segundos


# Listas para almacenar los valores de f(t) y g(t)
f_values = []
g_values = []
time_values = []

# Listas para almacenar los valores de velocidad
v_values_1 = []
v_values_2 = []

def update(frame):
    # tiempo en segundos
    t = (frame / fps) + tiempo_inicial  # tiempo inicial en segundos
    time_values.append(t)


    # --- Aceleración 1 ---
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
    

    # --- Velocidad 1 ---
    def v1(s):
        return v01 + quad(f, 0, s)[0]  # Integrar f(s) desde 0 hasta s
    
    # --- Velocidad 2 ---
    def v2(s):
        return v02 + quad(g, 0, s)[0]  # Integrar g(s) desde 0 hasta s
    

    # desplazamiento total en x
    dx1 = v01 * t + x_of1(t)
    dx2 = v02 * t + x_of2(t)

    # --- Coordenadas en cartesiano ---
    x1 = r1 * np.cos(theta1ord) + dx1
    y1 = r1 * np.sin(theta1ord)

    x2 = r2 * np.cos(theta2ord) + dx2
    y2 = r2 * np.sin(theta2ord)

    # --- Convertir de nuevo a polares ---
    r1_new = np.sqrt(x1**2 + y1**2)
    theta1_new = np.arctan2(y1, x1)
    r2_new = np.sqrt(x2**2 + y2**2)
    theta2_new = np.arctan2(y2, x2)

    # --- Expansión radial ---
    def expansion_factor1(time):
        return time**4# * 0.3 + 1  # Ajusta los parámetros
    def expansion_factor2(time):
        return  time**4# * 0.3 + 1  # Ajusta los parámetros
    
    r11 = r1_new*t + (expansion_factor1(t))
    r22 = r2_new*t + (expansion_factor2(t))
    

    # Generar puntos aleatorios dentro de las curvas
    num_puntos = 500
    x_puntos = []
    y_puntos = []

    for _ in range(num_puntos):
        # Curva 1
        theta = np.random.uniform(theta1_new.min(), theta1_new.max())
        r = np.random.uniform(0, np.interp(theta, theta1_new, r11))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x_puntos.append(x)
        y_puntos.append(y)

        # Curva 2
        theta = np.random.uniform(theta2_new.min(), theta2_new.max())
        r = np.random.uniform(0, np.interp(theta, theta2_new, r22))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x_puntos.append(x)
        y_puntos.append(y)

    x_puntos = np.array(x_puntos)
    y_puntos = np.array(y_puntos)

    # Crear el mapa de densidad
    heatmap, xedges, yedges = np.histogram2d(x_puntos, y_puntos, bins=num_bins, range=[[-50, 50], [-50, 50]])
    heatmap = gaussian_filter(heatmap, sigma=2)

    # Actualizar la imagen del mapa de densidad
    img.set_array(heatmap.T)

    # actualizar la curva
    line1.set_data(theta1_new, r11)
    line2.set_data(theta2_new, r22)
    ax.set_title(f"t = {t:.1f} s")

    # Almacenar los valores de f(t) y g(t)
    f_values.append(f(t))
    g_values.append(g(t))

    # Almacenar los valores de velocidad
    v_values_1.append(v1(t))
    v_values_2.append(v2(t))
    return line1, line2, img

     

# Animación
ani = FuncAnimation(fig, update, frames=300, init_func=init, blit=False)

out_path = "curva polar densidad.gif"
ani.save(out_path, writer=PillowWriter(fps=fps), dpi=200)

plt.close(fig)

out_path