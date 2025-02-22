import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
sns.set()
sns.set_context("talk")

# Definir el sistema de ecuaciones del atractor de Lorenz
def Lorenz(t, f, sigma, rho, beta):
    x, y, z = f
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])

def solveLorenz(T0, TF, iniciales, dt, sigma, rho, beta):
    T = np.arange(T0, TF + dt, dt)
    sol = solve_ivp(Lorenz,
                    t_span = [T.min(), T.max()],
                    t_eval = T,
                    y0 = iniciales,
                    args = (sigma, rho, beta),
                    method = 'BDF')
    return sol

def graficar(T0, TF, iniciales, dt, sigma, rho, beta):
    sol = solveLorenz(T0, TF, iniciales, dt, sigma, rho, beta)
    t, (x, y, z) = sol.t, sol.y
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(t, x, label='Conveccion (x)', linewidth=1.5, color='blue')
    ax.plot(t, y, label='Horizontal (y)', linewidth=1.5, color='darkorange')
    ax.plot(t, z, label='Vertical (z)', linewidth=1.5, color='green')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('u')
    #ax.set_title('Atractor de Lorenz')
    plt.legend()
    plt.tight_layout()
    plt.savefig("LorenzPython.pdf")
    plt.show()

def fase(T0, TF, iniciales, dt, sigma, rho, beta):
    sol = solveLorenz(T0, TF, iniciales, dt, sigma, rho, beta)
    t, (x, y, z) = sol.t, sol.y

    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel('Conveccion (x)')
    ax.set_ylabel('Horizontal (y)')
    #ax.set_title('Atractor de Lorenz (Espacio de fase x-y)')
    plt.tight_layout()
    plt.savefig("FaseXY_Python.pdf", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, z, linewidth=2)
    ax.set_xlabel('Conveccion (x)')
    ax.set_ylabel('Vertical (z)')
    #ax.set_title('Atractor de Lorenz (Espacio de fase x-z)')
    plt.tight_layout()
    plt.savefig("FaseXZ_Python.pdf", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y, z, linewidth=2)
    ax.set_xlabel('Horizontal (y)')
    ax.set_ylabel('Vertical (z)')
    #ax.set_title('Atractor de Lorenz (Espacio de fase y-z)')
    plt.tight_layout()
    plt.savefig("FaseYZ_Python.pdf", dpi=300)
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, linewidth=2)
    ax.set_xlabel('Conveccion (x)')
    ax.set_ylabel('Horizontal (y)')
    ax.set_zlabel('Vertical (z)')
    #ax.set_title('Espacio de fase del Atractor de Lorenz')
    plt.savefig("FaseXYZ_Python.pdf", dpi=300)
    plt.tight_layout()
    plt.show()


graficar(T0= 0., TF= 50., iniciales= [0, 1, 0], dt= 0.001, sigma= 10., rho= 28., beta= 2.5)
fase(T0= 0., TF= 50., iniciales= [0, 1, 0], dt= 0.001, sigma= 10., rho= 28., beta= 2.5)