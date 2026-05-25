# Creates a convergence plot of data produced by viscous_burgers/solve.py

"""
USAGE:
Run with the --sim arg to run a trial & generate data.
Run with the --plot arg to generate graphs (one of raw error, one of log vs log)
To choose the problem to simulate/plot, also pass in 'vb' or 'rb' for viscous burgers & reaction diffusion.
If you do not choose a problem, both will try to be solved and/or plotted
Run with --adaptive or --fixed to choose whether or not to use adaptive stepping or not (default: not)
"""

import matplotlib.pyplot as plt
import sys
import numpy as np
import math


args = sys.argv[1:]

k0 = 12

# Reference values
dx_ref = 1/20
dt_ref = 2 ** (-18)

k_f = 16 # Final dt value will be 2^(-k_f)

# t value to stop sim at
tf = 1

if "--adaptive" in args:
    adaptive = True
    method = "adaptive"
else:
    adaptive = False
    method = "fixed"

if "--sim" in args:
    def simulate(solver, problem_str):
        print("=== Generating reference ===")
        solver.solve(fname=f"{problem_str}_reference_{method}.csv", dx=dx_ref, dt=dt_ref, tf=tf, save_result=True, adaptive=False)
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
            result = solver.solve(fname=None, dx=dx, dt=dt, tf=tf, adaptive=adaptive) 
            # If this condition is false, then the solve blew up to infinity due to a large dt
            if not np.any(np.isnan(result)):
                dt_marker = np.array([dt]) # Put the inverse dt & the result together so they can be plotted later
                final_states.append(np.concatenate((dt_marker, result)))
            else:
                print("Error, dt is likely too large.")
    
        # Save trials results
        write_file(f"{problem_str}_trials_{method}.csv", final_states)

    if "vb" in args:
        from viscous_burgers import solve as vb
        simulate(vb, "vb")
    elif "rd" in args:
        from reaction_diffusion import solve as rd
        simulate(rd, "rd")
    else:
        from viscous_burgers import solve as vb
        from reaction_diffusion import solve as rd

        simulate(vb, "vb")
        simulate(rd, "rd")

if "--plot" in args:
    if "vb" in args:
        problems = ["vb"]
    elif "rd" in args:
        problems = ["rd"]
    else:
        problems = ["vb", "rd"]

    def get_ref_data(fname):
        ref = open(fname, "r") # Open file where reference data is stored. Should contain only one line.

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
    
    for p in problems:
        ref = get_ref_data(f"{p}_reference_{method}.csv")
        dt_values, results = get_trial_results(f"{p}_trials_{method}.csv")
        
        # Calculate relative L2 error for each trial (comparing to reference)
        dt_error = []
        for trial in results:
            diff = ref - trial
            error = np.max(diff) # Take the highest error value btwn the two discrete spaces
            dt_error.append(error)
        
        # Plot raw error values
        plt.figure()
        plt.axhline(y=0, color="orange", linestyle="--") # draw a zero line
        plt.plot(dt_values, dt_error)
        plt.xlabel("dt")
        plt.ylabel("error")
        plt.title(f"Raw error plot for {p} problem")

        plt.savefig(f"{p}_raw_error_{method}.png")

        # Plot log vs log
        log_dt = [math.log2(dt) for dt in dt_values]
        log_error = [math.log2(err) for err in dt_error]
        
        plt.figure()
        plt.plot(log_dt, log_error)

        # Reference lines
        x_ref = np.linspace(min(log_dt), max(log_dt), 100)
        y_slope_4 = 4 * x_ref
        y_slope_3 = 3 * x_ref
        y_slope_2 = 2 * x_ref
        y_slope_1 = 1 * x_ref

        plt.plot(x_ref, y_slope_4, "--", color="red", label="Slope = 4")
        plt.plot(x_ref, y_slope_3, "--", color="orange", label="Slope = 3")
        plt.plot(x_ref, y_slope_2, "--", color="yellow", label="Slope = 2")
        plt.plot(x_ref, y_slope_1, "--", color="green", label="Slope = 1")

        plt.xlabel("log2(dt)")
        plt.ylabel("log2(error)")
        plt.title(f"Convergence plot for {p} problem")

        plt.savefig(f"{p}_convergence_{method}.png")


