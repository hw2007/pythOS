# Creates a convergence plot of data produced by viscous_burgers/solve.py
# To run the simulation & generate data, pass the --sim argument. To use the data to create the plot, pass --plot.

import solve as vb
import matplotlib.pyplot as plt
import os
import sys


args = sys.argv[1:]

# Worst values to try
size_0 = 30
dt_0 = 200 # will actually be 1/dt_0

# Best values to try (should be <= to reference values)
size_f = 600
dt_f = 10000 # will actually be 1/dt_f

size_step = 30
dt_step = 200

if "--sim" in args:
    final_states = []

    os.makedirs("dt_study", exist_ok=True)
    os.makedirs("size_study", exist_ok=True)

    print("=== Beginning time study ===")

    size = size_f
    for dt_inverse in range(dt_0, dt_f+1, dt_step):
        dt = 1/dt_inverse
        print(f"Trial with size={size}, dt={dt}")
        try:
            vb.solve(fname=f"dt_study/{dt_inverse}.csv", size=size, dt=dt, tf=5)
        except:
            print("Error, values likely too small.")

    print("=== Beginning space size study ===")

    dt = 1/dt_f
    for size in range(size_0, size_f+1, size_step):
        print(f"Trial with size={size}, dt={dt}")
        try:
            vb.solve(fname=f"size_study/{size}.csv", size=size, dt=dt, tf=5)
        except:
            print("Error, values likely too small.")

if "--plot" in args:
    pass