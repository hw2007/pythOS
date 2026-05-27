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


# -(1 + e^-t) * xy * delta u
# delta u (laplacian) is expected to be pre-computed
def diffusion(t, x, y, lap):
    solution = -(1 + math.exp(-t)) * x * y * lap
    return solution


# Very long arbitrary equation
# Derived from the formulae given in Portero 2012 paper
# Derived using compute_f.py (simplified solution)
def f(t, x, y):
    pi = math.pi
    # Gross equation
    solution = 3*(2*pi*t*x*y*((2*pi*x**2*math.sin(pi*x**2) - math.cos(pi*x**2))*math.sin(pi*y)**2 - pi*math.sin(pi*x**2)*math.cos(2*pi*y))*(math.exp(t) + 1) + (1 - 2*t)*math.exp(t)*math.sin(pi*x**2)*math.sin(pi*y)**2)*math.exp(1 - 4*t)

    return solution


def subdomain1(t, u_vec):
    u = grid(u_vec)
    lap = laplacian(u)


operators = [diffusion, reaction]
methods = ["RK3", "RK3"]

# A guess dt value to start from
dt = 0.01

# Solve !!!
result = ark.ark_solve(operators, dt, y0, 0, T, methods, fname="results.csv", rtol=1e-2, atol=1e-3)
print(result)