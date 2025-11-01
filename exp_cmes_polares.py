import numpy as np
import matplotlib.pyplot as plt

# funciones

def velocidad_radial(theta, r, t):
    """
    Componente radial de la velocidad que evoluciona con el tiempo
    t: parámetro temporal (0 a 7 para los 8 estados)
    """
    fase = 2*np.pi*t/8
    return r * (1 - r) * (1 + 0.3*np.sin(fase))

def velocidad_angular(theta, r, t):
    """
    Componente angular de la velocidad que evoluciona con el tiempo
    """
    fase = 2*np.pi*t/8
    return np.sin(2*theta + fase) * (2 - r**2) * (1 + 0.2*np.cos(fase))

def densidad(theta, r, t):
    """
    Densidad que se propaga y cambia con el tiempo
    """
    fase = 2*np.pi*t/8
    # Densidad que se expande desde el centro
    onda = np.sin(3*r - fase) * 0.3
    angular = 0.2*np.cos(3*theta + fase)
    base = np.exp(-r/2)
    return base * (1 + onda + angular)

def cardioide_desplazado(theta, t, r0=0, theta0=0):
    """
    Genera un cardioide que se propaga en el plano polar
    t: tiempo (0-7)
    r0, theta0: desplazamiento inicial
    """
    # Desplazamiento radial que crece con el tiempo
    desplazamiento = 1.2 * t
    
    # CME básico
    r_base = 1 + (np.cos(theta))
    
    # Aplicar desplazamiento
    r = r_base + desplazamiento + r0
    
    return r

# ============================================================================
# CONFIGURACIÓN DE LA FIGURA CON 8 SUBPLOTS
# ============================================================================

fig = plt.figure(figsize=(20, 10))
fig.suptitle('Propagación de CME - Evolución de Densidad y Campo Vectorial', 
             fontsize=16, fontweight='bold', y=0.98)

# Crear 10 subplots en 2 filas x 5 columnas
estados_tiempo = 10

for idx in range(estados_tiempo):
    t = idx  # Tiempo discreto
    
    # Crear subplot polar
    ax = plt.subplot(2, 5, idx+1, projection='polar')
    
    # ========================================================================
    # CARDIOIDE EN POSICIÓN DESPLAZADA
    # ========================================================================
    theta_curva = np.linspace(0, 2 * np.pi, 10000)
    
    # Cada cardioide en una dirección diferente del plano polar
    angulo_propagacion = 0
    r_cardioide = cardioide_desplazado(theta_curva, t, theta0=angulo_propagacion)
    
    # ========================================================================
    # MALLA PARA DENSIDAD (dentro del cardioide desplazado)
    # ========================================================================
    theta_dense = np.linspace(0, 2 * np.pi, 200)
    r_dense = np.linspace(0, 3.5, 120)
    THETA_dense, R_dense = np.meshgrid(theta_dense, r_dense)
    
    # Calcular límite del cardioide desplazado
    R_limite_dense = cardioide_desplazado(THETA_dense, t, theta0=angulo_propagacion)
    mascara_dense = R_dense <= R_limite_dense
    
    # Calcular densidad evolutiva
    dens = densidad(THETA_dense, R_dense, t)
    dens_enmascarada = np.where(mascara_dense, dens, np.nan)
    
    # Graficar mapa de densidad
    contour = ax.contourf(THETA_dense, R_dense, dens_enmascarada, 
                          levels=20, cmap='YlOrRd', alpha=0.85)
    
    # ========================================================================
    # CAMPO VECTORIAL (menos denso para claridad)
    # ========================================================================
    theta_vec = np.linspace(0, 2 * np.pi, 16)
    r_vec = np.linspace(0.2, 3.0, 10)
    THETA_vec, R_vec = np.meshgrid(theta_vec, r_vec)
    
    # Límite para vectores
    R_limite_vec = cardioide_desplazado(THETA_vec, t, theta0=angulo_propagacion)
    mascara_vec = R_vec <= R_limite_vec
    
    # Calcular velocidades evolutivas
    v_r = velocidad_radial(THETA_vec, R_vec, t)
    v_theta = velocidad_angular(THETA_vec, R_vec, t)
    
    # Enmascarar
    v_r_masked = np.where(mascara_vec, v_r, np.nan)
    v_theta_masked = np.where(mascara_vec, v_theta, np.nan)
    
    # Convertir a componentes cartesianas
    U = v_r_masked * np.cos(THETA_vec) - v_theta_masked * np.sin(THETA_vec)
    V = v_r_masked * np.sin(THETA_vec) + v_theta_masked * np.cos(THETA_vec)
    
    # Dibujar vectores
    ax.quiver(THETA_vec, R_vec, U, V, 
              scale=20, width=0.003, color='black', alpha=0.7)
    
    # ========================================================================
    # CONTORNO DEL CARDIOIDE
    # ========================================================================
    #ax.plot(theta_curva, r_cardioide, 'darkred', linewidth=2.5)
    
    # ========================================================================
    # CONFIGURACIÓN DEL SUBPLOT
    # ========================================================================
    ax.set_title(f't = {t}', 
                 fontsize=11, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim([0, 3.5])
    
    # Añadir información de densidad promedio
    dens_promedio = np.nanmean(dens_enmascarada)
    ax.text(0.02, 0.98, f'ρ̄ = {dens_promedio:.3f}', 
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.7))

# ========================================================================
# BARRA DE COLOR GLOBAL PARA DENSIDAD
# ========================================================================
# Ajustar el layout para hacer espacio para la barra de color
plt.tight_layout(rect=[0, 0, 0.92, 0.96])

# Crear un eje para la barra de color a la derecha de todos los subplots
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contour, cax=cbar_ax)
cbar.set_label('Densidad ρ(θ,r,t)', rotation=270, labelpad=25, fontsize=13, fontweight='bold')
cbar.ax.tick_params(labelsize=10)
plt.savefig("py_py.pdf")
plt.show()

