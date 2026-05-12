# Equation: viscous Burgers, u_t = ν u_xx - 1/2 * (u^2)_x

import fractional_step as fs
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import random

SIZE = 500
LIMITS = (-6, 6)
dx = (LIMITS[1] - LIMITS[0]) / (SIZE-1)

# Viscosity coefficient ν
VISC = 0.1

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
tf = 5
num_steps = int(tf / dt)

# VISC * u_xx
def viscosity(t, y):
    dydt = np.zeros(SIZE)
    # dx^2 is used when computing second derivative
    dydt[1:-1] = VISC * (y[2:] - 2*y[1:-1] + y[:-2]) / dx**2
    
    return dydt

# -1/2 * (u^2)_x
def convection(t, y):
    dydt = np.zeros(SIZE)
    convect = -0.5 * y**2
    # Using 2 * dx because we are comparing points that are 2 cells steps away
    dydt[1:-1] = (convect[2:] - convect[:-2]) / (2*dx)

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

def save_animation():
    # Load all rows from CSV
    with open("results.csv", "r") as f:
        rows = list(f)

    fig, ax = plt.subplots()

    line, = ax.plot([], [], lw=2)

    ax.set_xlim(LIMITS[0], LIMITS[1])
    ax.set_ylim(0, 1)

    ax.set_xlabel("x")
    ax.set_ylabel("u")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame_idx):
        row = rows[frame_idx]

        data = row.strip().split(",")
        t = float(data[0])
        y_vals = np.array([float(v) for v in data[1:]])

        line.set_data(x, y_vals)
        ax.set_title(f"Discrete Space at t = {t:.3f}")

        return (line,)

    frames = range(0, len(rows), num_steps//(30*20)) # Animation will take 20 seconds

    animation = anim.FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
    )

    animation.save("animation.gif", writer="pillow", fps=30)

    plt.close()

operators = [viscosity, convection]
methods = {
    (1,): "RK3",
    (2,): "RK3"
}

print("Beginning solve...")

# Solve !!!
result = fs.fractional_step(operators, dt, y0, t0, tf, "Strang", methods, fname="results.csv")
print("DONE! Plotting graphs...")

# PLOTTING
snapshot(0)
snapshot(num_steps//4)
snapshot(num_steps//2)
snapshot(num_steps//4*3)
snapshot(num_steps)
print("Creating animation...")
save_animation()