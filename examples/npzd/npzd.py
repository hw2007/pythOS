# Equation: reaction-diffusion, u_t = D delta u + f(u)

import fractional_step as fs
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import time
from math import *


Nx = 100 # No. points in space
# Min and max values of space
DOMAIN = (-100, 0)

dx = (DOMAIN[1] - DOMAIN[0]) / Nx

# Diffusion coefficient kappa (using K instead of actual kappa for ease of typing)
K = 1e-2

# Configure timestepping
dt = 0.5
t0 = 0
tf = 5

num_steps = int(tf / dt)

# NPZD Params
u = 0.8
mP = 0.05
g = 1.2
kN = 0.3
kP = 0.3
B = 0.7
mZ = 0.03
rD = 0.04

# NPZD Init
x = np.array([DOMAIN[0] + (k + 0.5) * dx for k in range(Nx)])

N = [2.0 + 0.3*sin(2*pi*(k/Nx)) for k in range(0,Nx)]
P = [0.15 + 0.07*exp(x[k]/25) for k in range(0,Nx)]
Z = [0.10 + 0.02*exp(x[k]/25) for k in range(0,Nx)]
D = [0.05 for k in range(0,Nx)]

c0 = np.concatenate([N, P, Z, D])


def slice(c):
    """
    Takes a 1D np.array, c, and splits it into 4 quarters and outputs the quarters (as lists).
    """
    i = len(c)//4
    N = c[0:i]
    P = c[i:i*2]
    Z = c[i*2:i*3]
    D = c[i*3:i*4]

    return N, P, Z, D


def diffusion(t, c):
    slices = slice(c)
    dSlices = []

    for u in slices:
        dudt = np.zeros_like(u)
        for k in range(1, len(u)-1):
            # dx^2 is used when computing second derivative
            dudt[k] = K * (u[k+1] - 2*u[k] + u[k-1]) / dx**2
        dSlices.append(dudt)
    
    dcdt = np.concatenate(dSlices)
    return dcdt


def reaction(t, c):
    N, P, Z, D = slice(c)

    uptake = u * P * N / (kN + N)

    grazing = g * Z * P / (kP + P)

    dN = -uptake
    dP = uptake

    dP -= grazing
    dZ = B * grazing
    dD = (1 - B) * grazing + mP * P

    mortZ = mZ * Z
    dZ -= mortZ
    dD += mortZ

    remin = rD * D
    dD -= remin
    dN += remin

    dcdt = np.concatenate([dN, dP, dZ, dD])

    return dcdt


# Plot one single state of the sim
def snapshot(idx, csv_file):
    # idx: timestep to snapshot
    # csv_file: file to pull data from
    # Get the data from a timestep index
    def get_snapshot(fname, idx):
        f = open(fname, 'r')
        row = list(f)[idx]
        data = row.strip().split(",")[1:] # First value is time, dont use that one
        f.close()

        return [float(i) for i in data]

    y_vals = get_snapshot(csv_file, idx)

    # Plot it!pythOS-source/examples/npzd
    plt.figure()
    plt.plot(x, y_vals)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.ylim(0, 1)
    plt.title(f"Discrete Space at t = {idx / num_steps * (TF)}")

    plt.savefig(f"graph_{idx}.png")

def save_animation(csv_file):
    # Animate through the whole simulation in a gif
    # csv_file: file to pull data from

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


def plot(fname="results.csv"):
    # PLOTTING
    print("Plotting...")
    snapshot(0, fname)
    snapshot(num_steps//4, fname)
    snapshot(num_steps//2, fname)
    snapshot(num_steps//4*3, fname)
    snapshot(num_steps, fname)
    print("Creating animation...")
    save_animation(fname)

# Solve !!!
operators = [diffusion, reaction]
methods = {
    (1,): "RK3",
    (2,): "RK3"
}

start_time = time.perf_counter()
print("Beginning solve...")

result = fs.fractional_step(operators, dt, c0, t0, tf, "Strang", methods, fname="results.csv")

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")

print(f"Sum = {sum(result)}")
print(f"Min = {min(result)}")