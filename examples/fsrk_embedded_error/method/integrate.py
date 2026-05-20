# Equation: dy/dt = y

# Pass --result-only to only save the final result (not each step in time)

import fractional_step as fs
import math
import time
import sys
import rk4

def solve(t0, tf, dt, fname="results.csv"):
    # Initial condition
    y0 = 1

    def operator1(t, y):
        dydt = y
        return dydt

    operators = [operator1]
    methods = {
        (0,): rk4.rk4_method
    }

    result = fs.fractional_step(operators, dt, y0, t0, tf, "Strang", methods, fname=fname)
    
    return result

    
if __name__ == "__main__":
    print("BEGINNING SOLVE!")
    start_time = time.perf_counter()
    solve(0, 10, 1/10000, fname="results.csv")
    end_time = time.perf_counter()
    print(f"DONE! Solved in {end_time - start_time} seconds.")