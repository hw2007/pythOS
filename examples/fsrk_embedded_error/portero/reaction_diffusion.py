# Equation: reaction-diffusion, u_t = D delta u + f(u)

import additive_rk as ark
import matplotlib.pyplot as plt
import numpy as np
import time


GRID_SIZES = 16 # Number of grid points in each dimension.

T = 500 # Final time

# Domain of space
DOMAIN_X = (0, 1)
DOMAIN_Y = (0, 1)

# D * delta u
def diffusion(t, y):
    dydt = np.zeros(SIZE)
    # dx^2 is used when computing second derivative
    dydt[1:-1] = D * (y[2:] - 2*y[1:-1] + y[:-2]) / DX**2
    
    return dydt

# u(1 - u)
def reaction(t, y):
    dydt = np.zeros(SIZE)
    dydt = y * (1 - y)

    return dydt


operators = [diffusion, reaction]
methods = ["RK3", "RK3"]

# A guess dt value to start from
dt = 0.01

# Solve !!!
result = ark.ark_solve(operators, dt, y0, 0, T, methods, fname="results.csv", rtol=1e-2, atol=1e-3)
print(result)