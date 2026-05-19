# Creates a convergence plot of data produced by viscous_burgers/solve.py
# To run the simulation & generate data, pass the --sim argument. To use the data to create the plot, pass --plot.

import matplotlib.pyplot as plt
import sys
import numpy as np
import math


args = sys.argv[1:]

k_f = 18 # Final dt value will be 2^(-k_f)

# t value to stop sim at
tf = 10


if "--sim" in args:
    import integrate as solver

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
    for k in range(0, k_f+1):
        dt = 2 ** (-k)
        print(f"Trial with dt={dt}")
        # Solve PDE for this dt value
        result = solver.solve(0, tf, dt, fname=None) 
        # If this condition is false, then the solve blew up to infinity due to a large dt
        if result != None:
            final_states.append([k, result])
        else:
            print("Error, dt is likely too large.")
    
    # Save trials results
    write_file("trials.csv", final_states)
    print("All done!")

if "--plot" in args:
    def get_trial_results(fname):
        # Returns dt values, results

        trial = open(fname, "r") # Open file where trial data is stored

        k_vals = [] # values of k ,e.g. dt = 2^(-k)
        values = []
        for t in trial:
            trial_strings = t.rstrip().split(",") # Get each entry in the line
            trial_floats = [float(i) for i in trial_strings] # Convert to floats
            k_vals.append(trial_floats[0])
            values.append(trial_floats[1])
        
        return k_vals, values
    
    k_vals, values = get_trial_results("trials.csv")
    
    # Calculate error for each trial
    error = []
    for val in values:
        exact = math.exp(tf) # e^tf
        
        err = abs(val - exact)
        error.append(err)

    log_dt = [-i for i in k_vals]
    log_error = [math.log2(i) for i in error]
    
    # Plot it!
    plt.figure()
    plt.plot(log_dt, log_error)

    # Reference line: y = 4x
    x_ref = np.linspace(min(log_dt), max(log_dt), 100)
    y_ref = 4 * x_ref + 10

    plt.plot(x_ref, y_ref, "--", label="Slope = 4")

    plt.xlabel("log2(dt)")
    plt.ylabel("log2(error)")
    plt.title("Order plot for dy/dt = y")

    plt.savefig(f"order_graph.png")


