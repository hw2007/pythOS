# Creates a convergence plot of data produced by viscous_burgers/solve.py
# To run the simulation & generate data, pass the --sim argument. To use the data to create the plot, pass --plot.

import solve as vb
import matplotlib.pyplot as plt
import sys
import numpy as np


args = sys.argv[1:]

# Worst values to try
size_0 = 20
dt_0 = 10 # will actually be 1/dt_0

# Best values to try (should be <= to reference values)
size_f = 600
dt_f = 1000 # will actually be 1/dt_f

size_step = 20
dt_step = 10

if "--sim" in args:
    def write_file(fname, states):
        file = open(fname, "w")
        for state in states:
            string = state[0]
            for i in state[1:]:
                string = f"{string},{i}"
            file.write(string + "\n")
        file.close()

    print("=== Beginning time study ===")

    final_states = []
    size = size_f
    for dt_inverse in range(dt_0, dt_f+1, dt_step):
        dt = 1/dt_inverse
        print(f"Trial with size={size}, dt={dt}")
        result = vb.solve(fname=None, size=size, dt=dt, tf=5)
        if not np.any(np.isnan(result)):
            dt_marker = np.array([dt_inverse])
            final_states.append(np.concatenate((dt_marker, result)))
        else:
            print("Error, values likely too small.")
    
    write_file("dt_study.csv", final_states)

    print("=== Beginning space size study ===")

    final_states = []
    dt = 1/dt_f
    for size in range(size_0, size_f+1, size_step):
        print(f"Trial with size={size}, dt={dt}")
        result = vb.solve(fname=None, size=size, dt=dt, tf=5)
        if not np.any(np.isnan(result)):
            size_marker = np.array([size])
            final_states.append(np.concatenate((size_marker, result)))
        else:
            print("Error, values likely too small.")
    
    write_file("size_study.csv", final_states)

if "--plot" in args:
    pass