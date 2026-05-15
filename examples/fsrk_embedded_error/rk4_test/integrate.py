# Equation: dy/dt = y

import fractional_step as fs
import math
import time

# Start & end time
t0 = 0
tf = 10

# Timestep size
dt = 1/1000

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

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")