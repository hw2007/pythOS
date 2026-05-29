# Equation: reaction-diffusion, u_t = D delta u + f(u)

import additive_rk as ark
import butcher_tableau as bt
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import sys

# Define the method
c = np.array([1, 1, 4/9, 1/3])

A = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [-343/180, 5/9, 47/20, 0],
    [-1592/300, -121/60, 564/100, 47/20]
])

b_main = np.array([1/10, 1/4, 9/10, 3/4])
b_embedded = np.array([1, 1, 0, 0])

portero_2_1 = bt.EmbeddedTableau(c=c, a=A, b=b_main, b_aux=b_embedded, order=1)


N = 32 # Number of grid points in each dimension.

T = 500 # Final time

# Domain used for both x & y
DOMAIN = (0, 1)

dx = 1 / N

x_coords = np.linspace(DOMAIN[0], DOMAIN[1], N+1)
y_coords = np.linspace(DOMAIN[0], DOMAIN[1], N+1)
X, Y = np.meshgrid(x_coords, y_coords)


# Compute exact solution for a point
def exact_point(t, x, y):
    a = 3*t * math.exp(-3*t + 1)
    b = math.sin(math.pi * x**2)
    c = (math.sin(math.pi * y))**2
    return a*b*c


def full_exact_solution(t):
    u = np.zeros((N+1, N+1))

    for row in range(1, N):
        for col in range(1, N):
            x = X[row, col]
            y = Y[row, col]

            u[row,col] = exact_point(t, x, y)
    
    return u

u0 = full_exact_solution(0)


# Convert 2D grid -> 1D vector
def vec(u):
    return u.reshape((N+1) * (N+1))


# Convert 1D vector -> 2D grid
def grid(u):
    return u.reshape((N+1), (N+1))


def laplacian(u):
    lap = np.zeros_like(u)

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


# Partition of unity function
def part_of_unity(x, y):
    def h(z):
        return 2 ** (math.exp(2 - 1/z) / (2 * (z-1)))
    
    solution = 0

    if (x > 0 and x <= 3/16) or (x >= 9/16 and x <= 11/16):
        solution = 1
    elif (x >= 5/16 and x <= 7/16) or (x >= 13/16 and x < 1):
        solution = 0
    elif x > 3/16 and x < 5/16:
        solution = h(8*x - 3/2)
    elif x > 7/16 and x < 9/16:
        solution = 1 - h(8*x - 7/2)
    elif x > 11/16 and x < 13/16:
        solution = h(8*x - 11/2)
    """
    # Check if x is in subdomain 2
    if (x > 3/16 and x < 9/16) or (x > 11/16 and x < 1):
        solution = 1 - solution
    """
    return solution


chi1 = np.zeros_like(u0)
chi2 = np.zeros_like(u0)

for row in range(1, N):
    for col in range(1, N):
        x = X[row, col]
        y = Y[row, col]

        chi1[row, col] = part_of_unity(x, y)
        chi2[row, col] = 1 - part_of_unity(x, y)


def RHS(t, u_vec):
    u = grid(u_vec)
    dudt = np.zeros_like(u)

    lap = laplacian(u)

    for row in range(1, N):
        for col in range(1, N):
            x = X[row, col]
            y = Y[row, col]

            dudt[row,col] = diffusion(t, x, y) * lap[row,col] - u[row,col] + f(t, x, y)

    return dudt


def F1(t, u_vec):
    dudt = RHS(t, u_vec)
    dudt *= chi1

    return vec(dudt)

def F2(t, u_vec):
    dudt = RHS(t, u_vec)
    dudt *= chi2

    return vec(dudt)


args = sys.argv

if "--sim" in args:
    operators = [F1, F2]
    methods = [portero_2_1, portero_2_1]

    # A guess dt value to start from
    dt = 0.1

    print("BEGINNING SOLVE!")
    start_time = time.perf_counter()

    # Solve !!!
    result = ark.ark_solve(operators, dt, vec(u0), 0, T, methods, fname="results.csv", rtol=0, atol=dx**2)

    end_time = time.perf_counter()
    print(f"DONE! Solved in {end_time - start_time} seconds.")


def get_results(fname):
    f = open(fname, 'r')

    times = []
    values = []
    for line in f:
        data = line.rstrip().split(",")
        time = float(data[0])
        vals = [float(i) for i in data[1:]]
        times.append(time)
        values.append(vals)
    
    f.close()

    return times, values

if "--analyze" in sys.argv:
    times, values = get_results("results.csv")

    steps = []
    for i in range(1, len(times)):
        t0 = times[i-1]
        tf = times[i]
        steps.append(tf - t0)
    
    errors = []
    for i in range(len(values)):
        vals = values[i]
        t = times[i]

        # Check error
        exact_solution = full_exact_solution(t)
        exact_vec = vec(exact_solution)

        difference = np.abs(exact_vec - vals)
        error = np.sqrt(dx**2 * np.sum(difference**2))
        errors.append(error)

    average_step = sum(steps) / len(steps)
    print(f"Average step size = {average_step}")
    print(f"Global error = {max(errors)}")
