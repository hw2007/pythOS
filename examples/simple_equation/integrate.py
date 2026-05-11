# Equation: dy/dt = ay + bcos(t)

import fractional_step as fs
import math
import time

# Define parameters
a = 0.01
b = 5

# Start & end time
t0 = 0
tf = 300

# Timestep size
dt = 1/100

# Initial condition
y0 = 10 

# Perform 'ay'
def operator1(t, y):
    dydt = a * y
    return dydt

# Perform 'bcos(t)'
def operator2(t, y):
    dydt = b * math.cos(t)
    return dydt

operators = [operator1, operator2]
methods = {
    (0,): "", # Default
    (1,): "RK3",
    (2,): "RK3"
}

print("BEGINNING SOLVE!")
start_time = time.perf_counter()

result = fs.fractional_step(operators, dt, y0, t0, tf, "Strang", methods, fname="results.csv")

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")