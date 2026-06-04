from reaction_diffusion import solve as rd
from viscous_burgers import solve as vb
import methods

import sys, os, time
import numpy as np


args = sys.argv[1:]

# What methods are we testing?
M = [methods.portero_4_3, methods.portero_4_3]
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

rd_result = rd.solve(fname="rd_results.csv", N=N, dt=dt, tf=tf, methods=M)

end_time = time.perf_counter()
rd_time = end_time - start_time

print("Done, moving on...")

if not os.path.exists("vb_ref.csv"):
    print("No viscous Burgers reference found! Generating...")
    vb.solve(fname="vb_ref.csv", N=N, dt=dt_ref, tf=tf, methods=M_ref, save_result=True)

print("=== Beginning viscous Burgers test ===")
start_time = time.perf_counter()

vb_result = vb.solve(fname="vb_results.csv", N=N, dt=dt, tf=tf, methods=M)

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

f = open("rd_ref.csv", "r")
rd_ref = string_to_array(list(f)[0])
f.close()

f = open("vb_ref.csv", "r")
vb_ref = string_to_array(list(f)[0])
f.close()

rd_error = compare_solutions(rd_ref, rd_result)
vb_error = compare_solutions(vb_ref, vb_result)

print(f"Problem             | Compute Time (s)   | Error")
print("-----------------------------------------------------------")
print(f"Reaction-diffusion  | {rd_time} | {rd_error}")
print(f"Viscous Burgers     | {vb_time} | {vb_error}")