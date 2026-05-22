# Creates a convergence plot of data produced by viscous_burgers/solve.py
# To run the simulation & generate data, pass the --sim argument. To use the data to create the plot, pass --plot.

import matplotlib.pyplot as plt
import sys
import numpy as np


args = sys.argv[1:]

k0 = 8

# Reference values
dx_ref = 1/20
dt_ref = 2 ** (-20)

k_f = 16 # Final dt value will be 2^(-k_f)

# t value to stop sim at
tf = 1

if "--sim" in args:
    if "vb" in args:
        from viscous_burgers import solve as solver
    elif "rd" in args:
        from reaction_diffusion import solve as solver

    print("=== Generating reference ===")
    solver.solve(fname="reference.csv", dx=dx_ref, dt=dt_ref, tf=tf, save_result=True, adaptive=False)
    print("Done!")

    def write_file(fname, states):
        # Writes results of all trials to a file
        file = open(fname, "w")
        for state in states:
            string = state[0]
            for i in state[1:]:
                string = f"{string},{i}"
            file.write(string + "\n")
        file.close()

    print("=== Beginning trials ===")

    final_states = []
    dx = dx_ref
    for k in range(k0, k_f):
        dt = 2 ** (-k)
        print(f"Trial with dx={dx}, dt={dt}")
        # Solve PDE for this dt value
        result = solver.solve(fname=None, dx=dx, dt=dt, tf=tf) 
        # If this condition is false, then the solve blew up to infinity due to a large dt
        if not np.any(np.isnan(result)):
            dt_marker = np.array([dt]) # Put the inverse dt & the result together so they can be plotted later
            final_states.append(np.concatenate((dt_marker, result)))
        else:
            print("Error, dt is likely too large.")
    
    # Save trials results
    write_file("dt_trials.csv", final_states)

if "--plot" in args:
    def get_ref_data():
        ref = open("reference.csv", "r") # Open file where reference data is stored. Should contain only one line.

        # Extract reference data
        ref_data_strings = list(ref)[0].rstrip().split(",")
        ref.close()
        ref_data_floats = [float(i) for i in ref_data_strings]
        ref_data = np.array(ref_data_floats)

        return ref_data

    def get_trial_results(fname):
        # Returns dt values, results

        study = open(fname, "r") # Open file where study data is stored

        values = [] # Where the test values will be stored, e.g. dt for each trial
        results = []
        for t in study:
            trial_strings = t.rstrip().split(",") # Get each entry in the line
            trial_floats = [float(i) for i in trial_strings] # Convert to floats
            values.append(trial_floats[0]) # Get dt as float
            
            # Get the result of the trial
            trial_data = np.array(trial_floats[1:])
            results.append(trial_data)
        
        return values, results
    
    ref = get_ref_data()
    dt_values, results = get_trial_results("dt_trials.csv")
    
    # Calculate relative L2 error for each trial (comparing to reference)
    dt_error = []
    for trial in results:
        diff = ref - trial
        error = np.max(diff) # Take the highest error value btwn the two discrete spaces
        dt_error.append(error)
    
    # Plot it!
    plt.figure()
    plt.axhline(y=0, color="orange", linestyle="--") # draw a zero line
    plt.plot(dt_values, dt_error)
    plt.xlabel("dt")
    plt.ylabel("error")
    plt.title("Convergence plot for dt")

    plt.savefig(f"dt_plot.png")


