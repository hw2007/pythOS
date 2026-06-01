# Equation: reaction-diffusion, u_t = D delta u + f(u)

import additive_rk as ark
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import time
import methods as m


# Temporary values
dx = 1/32 # space between points. Passed to solve() and then set.

# Min and max values of space
LIMITS = (0, 1)
N = int((LIMITS[1] - LIMITS[0]) / dx) # No. of points in space

# Diffusion coefficient
D = 1

# Create wavefront
x = np.linspace(LIMITS[0], LIMITS[1], SIZE)
y0 = 1 / (1 + np.exp(x / np.sqrt(6)))**2

# Configure timestepping
t0 = 0
tf = 1

num_steps = int(tf / dt)

# D * delta u
def diffusion(t, c):
    dcdt = np.zeros(N)
    # dx^2 is used when computing second derivative
    for k in range(1, len(c)-1):
        dcdt[k] = D * (c[k+1] - 2*c[k] + c[k-1]) / dx**2
    
    return dcdt

# u(1 - u)
def reaction(t, y):
    dydt = np.zeros(SIZE)
    dydt = y * (1 - y)

    return dydt

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

    # Plot it!
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
methods = [m.heun_fe, m.sd2_be]

start_time = time.perf_counter()
print("Beginning solve...")

result = ark.ark_solve(operators, dt, y0, t0, tf, methods, fname="results.csv", rtol=1e-4, atol=1e-6)

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")
plot()