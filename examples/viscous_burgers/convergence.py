# Creates a convergence plot of data produced by viscous_burgers/solve.py
# To run the simulation & generate data, pass the --sim argument. To use the data to create the plot, pass --plot.

import matplotlib.pyplot as plt
import sys
import numpy as np


args = sys.argv[1:]

# IMPORTANT: All dx & dt values below must be give as inverses. So if you want a dx of 1/100, enter 100.

# Worst value to try
dt_0 = 100

dx_ref = 100
dt_ref = 100000

# Best value to try (should be <= to reference value)
dt_f = 5000

# t value to stop sim at
tf = 5

dt_step = 100

if "--sim" in args:
    import solve as vb

    print("=== Generating reference ===")
    vb.solve(fname="reference.csv", dx=1/dx_ref, dt=1/dt_ref, tf=tf, save_result=True)
    print("Done!")

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
    dx = 1/dx_ref
    for dt_inverse in range(dt_0, dt_f+1, dt_step):
        dt = 1/dt_inverse
        print(f"Trial with dx={dx}, dt={dt}")
        result = vb.solve(fname=None, dx=dx, dt=dt, tf=tf)
        if not np.any(np.isnan(result)):
            dt_marker = np.array([dt_inverse])
            final_states.append(np.concatenate((dt_marker, result)))
        else:
            print("Error, values likely too small.")
    
    write_file("dt_study.csv", final_states)

if "--plot" in args:
    def get_ref_data():
        ref = open("reference.csv", "r") # Open file where reference data is stored. Should contain only one line.

        # Extract reference data
        ref_data_strings = list(ref)[0].rstrip().split(",")
        ref.close()
        ref_data_floats = [float(i) for i in ref_data_strings]
        ref_data = np.array(ref_data_floats)

        return ref_data

    def get_study_data(fname):
        # Returns test values e.g. dt, results

        study = open(fname, "r") # Open file where study data is stored

        values = [] # Where the test values will be stored, e.g. dt for each trial
        study_data = []
        for t in study:
            trial_strings = t.rstrip().split(",")
            trial_floats = [float(i) for i in trial_strings]
            values.append(trial_floats[0])

            trial_data = np.array(trial_floats[1:])
            study_data.append(trial_data)
        
        return values, study_data
    
    ref = get_ref_data()
    dt_values, dt_study = get_study_data("dt_study.csv")

    dt_error = []
    for trial in dt_study:
        diff = ref - trial
        error = np.sqrt(np.sum(diff**2) * dx_ref)
        dt_error.append(float(error))
    
    # Plot it!
    plt.figure()
    plt.axhline(y=0, color="orange", linestyle="--")
    plt.plot(dt_values, dt_error)
    plt.xlabel("1/dt")
    plt.ylabel("Error")
    plt.title("Convergence plot for dt")

    plt.savefig(f"dt_plot.png")


