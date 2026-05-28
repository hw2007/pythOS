# Equation: reaction-diffusion, u_t = D delta u + f(u)

import additive_rk as ark
import matplotlib.pyplot as plt
import numpy as np
import time
import math


N = 16 # Number of grid points in each dimension.

T = 500 # Final time

u0_list = [[exact(0, x, y) for x in range(N)] for y in range(N)]
u0 = np.array(u0_list)

# Domain used for both x & y
DOMAIN = (0, 1)

dx = (DOMAIN[1] - DOMAIN[0]) / N


# Convert 2D grid -> 1D vector
def vec(u):
    return u.reshape(N * N)


# Convert 1D vector -> 2D grid
def grid(u):
    return u.reshape(N, N)


# Compute exact solution for a point
def exact(t, x, y):
    a = 3*t * math.exp(-3*t + 1)
    b = math.sin(math.pi * x**2)
    c = (math.sin(math.py * y))**2
    return a*b*c


def laplacian(u):
    lap = np.zeros_like(y)

    u_xx = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
    u_yy = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
    
    # Interior points only (not boundaries)
    
    lap[1:-1, 1:-1] = u_xx + u_yy
    
    return lap


# (1 + e^-t) * xy
def diffusion(t, x, y):
    solution = (1 + math.exp(-t)) * x * y
    return solution


# Very long arbitrary equation
# Derived from the formulae given in Portero 2012 paper
# Derived using compute_f.py (simplified solution)
def f(t, x, y):
    pi = math.pi
    # Gross equation
    solution = 3*(2*pi*t*x*y*((2*pi*x**2*math.sin(pi*x**2) - math.cos(pi*x**2))*math.sin(pi*y)**2 - pi*math.sin(pi*x**2)*math.cos(2*pi*y))*(math.exp(t) + 1) + (1 - 2*t)*math.exp(t)*math.sin(pi*x**2)*math.sin(pi*y)**2)*math.exp(1 - 4*t)

    return solution


def X1(x, y):
    def h(z):
        return 2 ** ((math.exp(2) - 1/z) / (2 * (z-1)))
    
    if (x > 0 and x <= 3/16) or (x >= 9/16 and x <= 11/16):
        return 1
    elif (x >= 5/16 and x <= 7/16) or (x >= 13/16 and x < 1):
        return 0
    elif x > 3/16 and x < 5/16:
        return h(8*x - 3/2)
    elif x > 7/16 and x < 9/16:
        return 1 - h(8*x - 7/2)
    elif x > 11/16 and x < 13/16:
        return h(8*x - 11/2)


def X2(x, y):
    return 1 - X1(x, y) 


def subdomain1(t, u_vec):
    u = grid(u_vec)
    lap = laplacian(u)

    dudt = zeros_like(u)

    for x in range(DOMAIN[0], DOMAIN[1], dx)
        dudt[y,x] = X1(x, y) * (diffusion(t, x, y) * lap + u[y,x]) + X1(x, y) * f(t, x, y)

    return dudt


operators = [diffusion, reaction]
methods = ["RK3", "RK3"]

# A guess dt value to start from
dt = 0.01

# Solve !!!
result = ark.ark_solve(operators, dt, y0, 0, T, methods, fname="results.csv", rtol=1e-2, atol=1e-3)
print(result)