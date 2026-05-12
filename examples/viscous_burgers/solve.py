# Equation: viscous Burgers, u_t = ν u_xx - 1/2 * (u^2)_x

import fractional_step as fs
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import random

SIZE = 100
LIMITS = (-6, 6)
dx = (LIMITS[1] - LIMITS[0]) / (SIZE-1)

# Viscosity coefficient ν
VISC = 0.01

"""
initial_list = []
random.seed = 314
for i in range(SIZE):
    initial_list.append(random.random())

y0 = np.array(initial_list)
"""

x = np.linspace(LIMITS[0], LIMITS[1], SIZE)
y0 = np.exp(-(x**2) / 2)

dt = 1/1000
t0 = 0
tf = 1
num_steps = int(tf / dt)

# VISC * u_xx
def viscosity(t, y):
    dydt = np.zeros(SIZE)
    dydt[1:-1] = VISC * (y[2:] - 2*y[1:-1] + y[:-2]) / dx**2
    
    return dydt

# -u * u_x
def convection(t, y):
    dydt = np.zeros(SIZE)
    flux = 0.5 * y**2
    dydt[1:-1] = -(flux[2:] - flux[:-2]) / (2*dx)

    return dydt

def snapshot(idx):
    def get_snapshot(fname, idx):
        f = open(fname, 'r')
        row = list(f)[idx]
        data = row.strip().split(",")[1:] # First value is time, dont use that one
        f.close()

        return [float(i) for i in data]

    y_vals = get_snapshot("results.csv", idx)

    # Plot it!
    plt.figure()
    plt.plot(x, y_vals)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.ylim(0, 1)
    plt.title(f"Discrete Space at t = {idx / num_steps * (tf-t0)}")

    plt.savefig(f"graph_{idx}.png")

operators = [viscosity, convection]
methods = {
    (1,): "RK3",
    (2,): "RK3"
}

print("Beginning solve...")

# Solve !!!
result = fs.fractional_step(operators, dt, y0, t0, tf, "Strang", methods, fname="results.csv")
print("DONE!")

# PLOTTING
snapshot(0)
snapshot(1)
snapshot(2)
snapshot(3)
snapshot(4)
snapshot(5)
snapshot(10)
snapshot(20)
snapshot(num_steps//4)
snapshot(num_steps//2)
snapshot(num_steps//4*3)
snapshot(num_steps)