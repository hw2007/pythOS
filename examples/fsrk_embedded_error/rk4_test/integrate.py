# Equation: dy/dt = y

# Pass --result-only to only save the final result (not each step in time)

import fractional_step as fs
import math
import time
import sys

args = sys.argv[1:]
if "--result-only" in args:
    fname = None
    result_only = True
    print("Only result will be saved!")
else:
    fname = "results.csv"
    result_only = False


# Start & end time
t0 = 0
tf = 10

# Timestep size
dt = 1/100000

# Initial condition
y0 = 10 

def operator1(t, y):
    dydt = y
    return dydt

operators = [operator1]
methods = {
    (0,): "", # Default
    (1,): "RK4",
    (2,): "RK4"
}

print("BEGINNING SOLVE!")
start_time = time.perf_counter()

result = fs.fractional_step(operators, dt, y0, t0, tf, "Strang", methods, fname="results.csv")

if result_only:
    file = open("results.csv", "w")
    file.write(str(result))
    file.close()

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")