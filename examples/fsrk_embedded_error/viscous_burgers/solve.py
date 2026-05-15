# Equation: viscous Burgers, u_t = ν u_xx - 1/2 * (u^2)_x

import fractional_step as fs
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


# Temporary values
SIZE = 0 # num of points in space. Gets calculated when solve() is called.
DX = 0 # space between points. Passed to solve() and then set.

TF = 1 # Final time. Will be set by solve()

# Min and max values of space
LIMITS = (-6, 6)

# Viscosity coefficient ν
VISC = 0.01

num_steps = 0 # Will be calculated when solve() is called

# VISC * u_xx
def viscosity(t, y):
    dydt = np.zeros(SIZE)
    # dx^2 is used when computing second derivative
    dydt[1:-1] = VISC * (y[2:] - 2*y[1:-1] + y[:-2]) / DX**2
    
    return dydt

# -1/2 * (u^2)_x
def convection(t, y):
    dydt = np.zeros(SIZE)
    convect = -0.5 * y**2
    # Using 2 * dx because we are comparing points that are 2 cells steps away
    dydt[1:-1] = (convect[2:] - convect[:-2]) / (2*DX)

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

def solve(fname="results.csv", dx=1/100, dt=1/4000, tf=5, save_result=False):
    # Solve the PDE!
    # save_result: if True, only save the last step in the simulation. Otherwise save everything.
        
    global SIZE, DX, x, num_steps, TF

    if save_result: filename = None
    else: filename = fname

    # Configure discrete space
    DX = dx
    SIZE = int((LIMITS[1] - LIMITS[0]) / DX)

    # Create gaussian bump
    x = np.linspace(LIMITS[0], LIMITS[1], SIZE)
    y0 = np.exp(-(x**2) / 2)
    
    # Configure timestepping
    t0 = 0
    TF = tf
    num_steps = int(tf / dt)

    operators = [convection, viscosity]
    methods = {
        (1,): "RK3",
        (2,): "RK3"
    }

    # Solve !!!
    result = fs.fractional_step(operators, dt, y0, t0, tf, "Strang", methods, fname=filename)
    
    if save_result:
        file = open(fname, "w")
        string = result[0]
        for i in result[1:]:
            string = f"{string},{i}"
        file.write(string + "\n")
        file.close()

    return result

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

if __name__ == "__main__":
    print("Beginning solve...")
    solve()
    print("DONE!")
    plot()