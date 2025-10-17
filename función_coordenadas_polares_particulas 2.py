import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.integrate import quad
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
#########################################################
fps = 20

fig = plt.figure()
ax = plt.subplot(111, polar=True)  # Crear un gráfico polar
ax.set_rmax(1000)  # radio maximo del plano polar
ax.set_thetamin(90)  # límite inferior en grados, para que se muestre solamente la mitad del plano.
ax.set_thetamax(-90)  # límite superior en grados
circle = Circle((0, 0), 0.1, transform=ax.transData._b, color="red", alpha=1)
ax.add_patch(circle)
line1, = ax.plot([], [], lw=2, color="blue", label="Curva 1")
line2, = ax.plot([], [], lw=2, color="red", label="Curva 2")
scatter1 = ax.scatter([], [], s=2, color="blue", alpha=0.5)
scatter2 = ax.scatter([], [], s=2, color="red", alpha=0.5)
scatter11 = ax.scatter([], [], s=2, color="green", alpha=0.5)
scatter22 = ax.scatter([], [], s=2, color="purple", alpha=0.5)
scatter111 = ax.scatter([], [], s=2, color="orange", alpha=0.5)
scatter222 = ax.scatter([], [], s=2, color="brown", alpha=0.5)


# para que las curvas se desplacen:
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    scatter1.set_offsets(np.array([[], []]).T)
    scatter2.set_offsets(np.array([[], []]).T)
    scatter11.set_offsets(np.array([[], []]).T)
    scatter22.set_offsets(np.array([[], []]).T)
    scatter111.set_offsets(np.array([[], []]).T)
    scatter222.set_offsets(np.array([[], []]).T)
    return line1, line2, scatter1, scatter2, scatter11, scatter22, scatter111, scatter222

# parametros para la aceleración

# Para la CME1
tr1 = 0.3
td1 = 1
ar1 = 0.1
ad1 = 2
v01 = 0.2
# Para la CME2
tr2 = 0.2
td2 = 5
ar2 = 0.1
ad2 = 3
v02 = 0

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


#Expansión radial
def expansion_factor1(time):
    return time**4
def expansion_factor2(time):
    return time**6

tiempo_inicial = 0  # tiempo inicial en segundos
############################################################################################
# Listas para almacenar los valores de f(t) y g(t)
f_values = []
g_values = []
time_values = []
# Listas para almacenar los valores de velocidad
v_values_1 = []
v_values_2 = []

# NUEVAS LISTAS PARA r11 CON theta11 = 0
r11_theta0_values = []  # Posición de r11 cuando theta11 = 0
v_r11_theta0_values = []  # Velocidad de r11 cuando theta11 = 0
a_r11_theta0_values = []  # Aceleración de r11 cuando theta11 = 0


# NUEVAS LISTAS PARA r22 CON theta22 = 0
r22_theta0_values = []  # Posición de r22 cuando theta22 = 0
v_r22_theta0_values = []  # Velocidad de r22 cuando theta22 = 0
a_r22_theta0_values = []  # Aceleración de r22 cuando theta22 = 0
########################################################################################

theta1 = []
theta2 = []
r1 = []  # Radio 1 en función de θ
r2 = []  # Radio 2 en función de θ

for i in range(100):
    r1.append(rd.uniform(2, 2.05))
    theta1.append(rd.uniform(0, 2 * np.pi))
    r2.append(rd.uniform(1, 1.055))
    theta2.append(rd.uniform(0, 2 * np.pi))

theta1ord = np.append(sorted(theta1), sorted(theta1)[0])  # Ordenar los ángulos
r1 = np.append(r1, r1[0])
theta2ord = np.append(sorted(theta2), sorted(theta2)[0])
r2 = np.append(r2, r2[0])

def update(frame):
    # tiempo en segundos
    t = (frame / fps) + tiempo_inicial
    time_values.append(t)
    # --- Velocidad 1 ---
    def v1(s):
        return v01 + quad(f, 0, s)[0]  # Integrar f(s) desde 0 hasta s
    
    # --- Velocidad 2 ---
    def v2(s):
        return v02 + quad(g, 0, s)[0]  # Integrar g(s) desde 0 hasta s
    
    # desplazamiento total en x
    dx1 = v01 * t + x_of1(t)
    dx2 = v02 * t + x_of2(t)
    #######Para las curvas externas######
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
    r11 = (r1_new * t) + (expansion_factor1(t))
    r22 = (r2_new * t) + (expansion_factor2(t))
    # --- Actualizar las posiciones de las partículas en coordenadas polares ---
    scatter1.set_offsets(np.column_stack((theta1_new, r11)))
    scatter2.set_offsets(np.column_stack((theta2_new, r22)))

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

    # --- CALCULAR POSICIÓN DEL FRENTE DE CME1 (theta ≈ 0°) ---
    idx1_theta0 = np.argmin(np.abs(theta1_new))
    r11_theta0 = r11[idx1_theta0]
    r11_theta0_values.append(r11_theta0)

    # --- CALCULAR POSICIÓN DEL FRENTE DE CME2 (theta ≈ 0°) ---
    idx2_theta0 = np.argmin(np.abs(theta2_new))
    r22_theta0 = r22[idx2_theta0]
    r22_theta0_values.append(r22_theta0)

    return line1, line2, scatter1, scatter2

######################################################################################

theta11 = []
theta22 = []
r11 = []  # Radio 1 en función de θ (cambiado el nombre para evitar conflicto)
r22 = []  # Radio 2 en función de θ

for i in range(100):
    r11.append(rd.uniform(1.5, 1.8))
    theta11.append(rd.uniform(0, 2 * np.pi))
    r22.append(rd.uniform(1.7, 1.9))
    theta22.append(rd.uniform(0, 2 * np.pi))

theta11ord = np.append(sorted(theta11), sorted(theta11)[0])  # Ordenar los ángulos
r11 = np.append(r11, r11[0])
theta22ord = np.append(sorted(theta22), sorted(theta22)[0])
r22 = np.append(r22, r22[0])

def update2(frame, Dr1, Dr2):
    # tiempo en segundos
    t = (frame / fps) + tiempo_inicial

    # desplazamiento total en x
    dx1 = v01 * t + x_of1(t)
    dx2 = v02 * t + x_of2(t)
    #######Para las curvas internas######
    # --- Coordenadas en cartesiano ---
    x1 = (r11 + Dr1) * np.cos(theta11ord) + dx1
    y1 = (r11 + Dr1) * np.sin(theta11ord)

    x2 = (r22 + Dr2) * np.cos(theta22ord) + dx2
    y2 = (r22 + Dr2) * np.sin(theta22ord)

    # --- Convertir de nuevo a polares ---
    r1_new = np.sqrt(x1**2 + y1**2)
    theta1_new = np.arctan2(y1, x1)
    r2_new = np.sqrt(x2**2 + y2**2)
    theta2_new = np.arctan2(y2, x2)
    # --- Expansión radial ---

    r1 = (r1_new * t) + (expansion_factor1(t))
    r2 = (r2_new * t) + (expansion_factor2(t))
    
    # --- Actualizar las posiciones de las partículas en coordenadas polares ---
    scatter11.set_offsets(np.column_stack((theta1_new, r1)))
    scatter22.set_offsets(np.column_stack((theta2_new, r2)))
    return scatter11, scatter22

################################################################################

theta111 = []
theta222 = []
r111 = []  # Radio 1 en función de θ
r222 = []  # Radio 2 en función de θ

for i in range(100):
    r111.append(rd.uniform(1.5, 1.8))
    theta111.append(rd.uniform(0, 2 * np.pi))
    r222.append(rd.uniform(1.7, 1.9))
    theta222.append(rd.uniform(0, 2 * np.pi))

theta111ord = np.append(sorted(theta111), sorted(theta111)[0])  # Ordenar los ángulos
r111 = np.append(r111, r111[0])
theta222ord = np.append(sorted(theta222), sorted(theta222)[0])
r222 = np.append(r222, r222[0])

def update3(frame, Dr1, Dr2):
    # tiempo en segundos
    t = (frame / fps) + tiempo_inicial

    # desplazamiento total en x
    dx1 = v01 * t + x_of1(t)
    dx2 = v02 * t + x_of2(t)
    #######Para las curvas internas######
    # --- Coordenadas en cartesiano ---
    x1 = (r111 + Dr1) * np.cos(theta111ord) + dx1
    y1 = (r111 + Dr1) * np.sin(theta111ord)

    x2 = (r222 + Dr2) * np.cos(theta222ord) + dx2
    y2 = (r222 + Dr2) * np.sin(theta222ord)

    # --- Convertir de nuevo a polares ---
    r1_new = np.sqrt(x1**2 + y1**2)
    theta1_new = np.arctan2(y1, x1)
    r2_new = np.sqrt(x2**2 + y2**2)
    theta2_new = np.arctan2(y2, x2)
    # --- Expansión radial ---
    r1 = (r1_new * t) + (expansion_factor1(t))
    r2 = (r2_new * t) + (expansion_factor2(t))
# --- Actualizar las posiciones de las partículas en coordenadas polares ---
    scatter111.set_offsets(np.column_stack((theta1_new, r1)))
    scatter222.set_offsets(np.column_stack((theta2_new, r2)))
    return scatter111, scatter222

###################################################################################

def updatefin(frame):
    update(frame)
    update2(frame, -1.5, -1.5)
    update3(frame, -1, -1)
    
    # Calcular velocidad y aceleración de r11 con theta11 = 0
    if len(r11_theta0_values) > 1:
        # Calcular velocidad (derivada numérica)
        dt = 1/fps
        current_velocity1 = (r11_theta0_values[-1] - r11_theta0_values[-2]) / dt
        v_r11_theta0_values.append(current_velocity1)
        
        # Calcular aceleración (derivada segunda numérica)
        if len(v_r11_theta0_values) > 1:
            current_acceleration1 = (v_r11_theta0_values[-1] - v_r11_theta0_values[-2]) / dt
            a_r11_theta0_values.append(current_acceleration1)
     # Calcular velocidad y aceleración de r22 con theta22 = 0
    if len(r22_theta0_values) > 1:
        # Calcular velocidad (derivada numérica)
        dt = 1/fps
        current_velocity2 = (r22_theta0_values[-1] - r22_theta0_values[-2]) / dt
        v_r22_theta0_values.append(current_velocity2)
        
        # Calcular aceleración (derivada segunda numérica)
        if len(v_r22_theta0_values) > 1:
            current_acceleration2 = (v_r22_theta0_values[-1] - v_r22_theta0_values[-2]) / dt
            a_r22_theta0_values.append(current_acceleration2)
    
    return line1, line2, scatter1, scatter2, scatter11, scatter22, scatter111, scatter222


# Animación
ani = FuncAnimation(fig, updatefin, frames=1000, init_func=init, blit=False)

out_path = "curva_polar_particulas.gif"
ani.save(out_path, writer=PillowWriter(fps=fps), dpi=200)

plt.close(fig)

# Crear una nueva figura para graficar f(t) y g(t)
plt.figure()
plt.plot(time_values, f_values, label='f(t)', color='blue')
plt.plot(time_values, g_values, label='g(t)', color='red')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración')
plt.title('Aceleración en función del tiempo')
plt.legend()
plt.savefig('aceleracion_vs_tiempo.png')
plt.grid(True)
plt.show()

# Crear una nueva figura para graficar las velocidades de la CME1 y CME2
plt.figure()
plt.plot(time_values, v_values_1, label='v1(t)', color='blue')
plt.plot(time_values, v_values_2, label='v2(t)', color='red')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad')
plt.title('Velocidad en función del tiempo')
plt.legend()
plt.savefig('velocidad_vs_tiempo.png')
plt.grid(True)
plt.show()

# NUEVA GRÁFICA: Velocidad y Aceleración de r11 con theta11 = 0
plt.figure(figsize=(12, 8))
# Subplot 1: Posición de r11 y r22
plt.subplot(3, 1, 1)
plt.plot(time_values[:len(r11_theta0_values)], r11_theta0_values, 'g-', linewidth=2, label='r11 (theta=0)')
plt.plot(time_values[:len(r22_theta0_values)], r22_theta0_values, 'm-', linewidth=2, label='r22 (theta=0)')
plt.ylabel('Posición')
plt.title('Movimiento de Puntos r11 y r22 (theta=0) - Posición, Velocidad y Aceleración')
plt.legend()
plt.grid(True)

# Subplot 2: Velocidad de r11 y r22
plt.subplot(3, 1, 2)
if len(v_r11_theta0_values) > 0:
    time_velocity = time_values[1:len(v_r11_theta0_values)+1]
    plt.plot(time_velocity, v_r11_theta0_values, 'r-', linewidth=2, label='v_r11')
if len(v_r22_theta0_values) > 0:
    time_velocity2 = time_values[1:len(v_r22_theta0_values)+1]
    plt.plot(time_velocity2, v_r22_theta0_values, 'c-', linewidth=2, label='v_r22')
plt.ylabel('Velocidad')
plt.legend()
plt.grid(True)

# Subplot 3: Aceleración de r11 y r22
plt.subplot(3, 1, 3)
if len(a_r11_theta0_values) > 0:
    time_accel1 = time_values[2:len(a_r11_theta0_values)+2]
    plt.plot(time_accel1, a_r11_theta0_values, 'b-', linewidth=2, label='a_r11')
if len(a_r22_theta0_values) > 0:
    time_accel2 = time_values[2:len(a_r22_theta0_values)+2]
    plt.plot(time_accel2, a_r22_theta0_values, 'y-', linewidth=2, label='a_r22')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('r11_y_r22_theta0_velocidad_aceleracion.png')
plt.show()

# Mostrar valores estadísticos
if len(r11_theta0_values) > 0:
    print(f"Análisis del punto r11 con theta11 = 0:")
    print(f"Posición máxima: {max(r11_theta0_values):.3f}")
    if len(v_r11_theta0_values) > 0:
        print(f"Velocidad máxima: {max(v_r11_theta0_values):.3f}")
    if len(a_r11_theta0_values) > 0:
        print(f"Aceleración máxima: {max(a_r11_theta0_values):.3f}")

if len(r22_theta0_values) > 0:
    print(f"\nAnálisis del punto r22 con theta22 = 0:")
    print(f"Posición máxima: {max(r22_theta0_values):.3f}")
    if len(v_r22_theta0_values) > 0:
        print(f"Velocidad máxima: {max(v_r22_theta0_values):.3f}")
    if len(a_r22_theta0_values) > 0:
        print(f"Aceleración máxima: {max(a_r22_theta0_values):.3f}")
out_path