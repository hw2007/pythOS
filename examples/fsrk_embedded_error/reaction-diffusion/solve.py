# Equation: reaction-diffusion, u_t = D delta u + f(u)

import fractional_step as fs
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import random

SIZE = 500
LIMITS = (-20, 20)
dx = (LIMITS[1] - LIMITS[0]) / (SIZE-1)

# Diffusion coefficient
D = 1

"""
initial_list = []
random.seed = 314
for i in range(SIZE):
    initial_list.append(random.random())

y0 = np.array(initial_list)
"""

x = np.linspace(LIMITS[0], LIMITS[1], SIZE)
#y0 = np.exp(-(x**2) / 2)
y0 = 1 / (1 + np.exp(x / np.sqrt(6)))**2

dt = 1/1000
t0 = 0
tf = 5
num_steps = int(tf / dt)

# D * delta u
def diffusion(t, y):
    dydt = np.zeros(SIZE)
    # dx^2 is used when computing second derivative
    dydt[1:-1] = D * (y[2:] - 2*y[1:-1] + y[:-2]) / dx**2
    
    return dydt

# u(1 - u)
def reaction(t, y):
    dydt = np.zeros(SIZE)
    dydt = y * (1 - y)

    return dydt

def snapshot(idx, csv_file):
    def get_snapshot(fname, idx):
        f = open(fname, 'r')
        row = list(f)[idx]
        data = row.strip().split(",")[1:] # First value is time, dont use that one
        f.close()

        return [float(i) for i in data]

    y_vals = get_snapshot(csv_file, idx)

    # Plot it!
    plt.figure()
    plt.plot(x, y_vals)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.ylim(0, 1)
    plt.title(f"Discrete Space at t = {idx / num_steps * (tf-t0)}")

    plt.savefig(f"graph_{idx}.png")

def save_animation(csv_file):
    # Load all rows from CSV
    with open(csv_file, "r") as f:
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

def solve(fname: str):
    operators = [reaction, diffusion]
    methods = {
        (1,): "RK3",
        (2,): "RK3"
    }

    print("Beginning solve...")

    # Solve !!!
    result = fs.fractional_step(operators, dt, y0, t0, tf, "Strang", methods, fname=fname)
    print("DONE!")

def plot(fname: str):
    # PLOTTING
    print("Plotting...")
    snapshot(0, fname)
    snapshot(num_steps//4, fname)
    snapshot(num_steps//2, fname)
    snapshot(num_steps//4*3, fname)
    snapshot(num_steps, fname)
    print("Creating animation...")
    save_animation(fname)

if __name__ == "__main__":
    solve("results.csv")
    plot("results.csv")