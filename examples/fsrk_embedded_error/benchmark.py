from reaction_diffusion import solve as rd
from viscous_burgers import solve as vb
import methods

import sys, os, time
import numpy as np


args = sys.argv[1:]

# What methods are we testing?
M = [methods.portero_3_2, methods.portero_3_2]
N = 1000
dt = 1e-3
tf = 5

# Used for reference solutions
dt_ref = 1e-5
M_ref = [methods.rk3, methods.rk3]

if not os.path.exists("rd_ref.csv"):
    print("No reaction-diffusion reference found! Generating...")
    rd.solve(fname="rd_ref.csv", N=N, dt=dt_ref, tf=tf, methods=M_ref, save_result=True)

print("=== Beginning reaction-diffusion test ===")
start_time = time.perf_counter()

rd_result, rd_rejects = rd.solve(fname="rd_results.csv", N=N, dt=dt, tf=tf, methods=M, track_rejects=True)

end_time = time.perf_counter()
rd_time = end_time - start_time

print("Done, moving on...")

if not os.path.exists("vb_ref.csv"):
    print("No viscous Burgers reference found! Generating...")
    vb.solve(fname="vb_ref.csv", N=N, dt=dt_ref, tf=tf, methods=M_ref, save_result=True)

print("=== Beginning viscous Burgers test ===")
start_time = time.perf_counter()

vb_result, vb_rejects = vb.solve(fname="vb_results.csv", N=N, dt=dt, tf=tf, methods=M, track_rejects=True)

end_time = time.perf_counter()
vb_time = end_time - start_time

print("Done, moving on...")

print("=== The results are in! ===")

def string_to_array(string):
    """
    Convert CSV string like "1,2,3,4" to array like [1 2 3 4]
    string must be made up comma-separated of float values
    """

    substrs = string.rstrip().split(",")
    arr = np.array([float(s) for s in substrs])

    return arr

def compare_solutions(ref, soln):
    """
    Compare soln to the ref
    Returns the maximal L2 error 
    ref & soln should be given as np.array
    """

    diff = ref - soln
    err_array = np.sqrt(np.mean(diff**2))

    error = np.max(err_array)
    return error

def get_timesteps(fname):
    """
    Take a result, and get a list of all timestep sizes
    """

    f = open(fname, "r")
    fdata = list(f)
    f.close()

    step_sizes = []

    for i in range(1, len(fdata)):
        prev_line = fdata[i-1]
        current_line = fdata[i]
        prev_time = float(prev_line.rstrip().split(",")[0])
        time = float(current_line.rstrip().split(",")[0])
        
        step = time - prev_time
        step_sizes.append(step)
    
    return step_sizes

f = open("rd_ref.csv", "r")
rd_ref = string_to_array(list(f)[0])
f.close()

f = open("vb_ref.csv", "r")
vb_ref = string_to_array(list(f)[0])
f.close()

rd_error = compare_solutions(rd_ref, rd_result)
vb_error = compare_solutions(vb_ref, vb_result)

rd_steps = get_timesteps("rd_results.csv")
vb_steps = get_timesteps("vb_results.csv")

rd_average_step = sum(rd_steps) / len(rd_steps)
vb_average_step = sum(vb_steps) / len(vb_steps)

print(f"Problem             | Compute Time (s)   | Error              | Avg Stepsize       | Reject/Accept")
print("------------------------------------------------------------------------------------------------------")
print(f"Reaction-diffusion  | {rd_time} | {rd_error} | {rd_average_step} | {rd_rejects / len(rd_steps)}")
print(f"Viscous Burgers     | {vb_time} | {vb_error} | {vb_average_step} | {vb_rejects / len(vb_steps)}")
