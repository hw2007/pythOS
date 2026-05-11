import fractional_step as fs
import time
import numpy as np

# Define parameters: mass of Earth & Moon
moon = 0.012277471
earth = 1 - moon

# Initial conditions
x0 = 0.994
x_prime0 = 0
y0 = 0
y_prime0 = -2.001585106

Y0 = np.array([x0, y0, x_prime0, y_prime0], dtype=float)

# Time
t0 = 0
tf = 17.06521656015796 * 2
dt = 1/5000

def drift(t, Y):
    x, y, x_prime, y_prime = Y

    return np.array([x_prime, y_prime, 0, 0], dtype=float)

def kick(t, Y):
    x, y, x_prime, y_prime = Y

    D1 = ((x + moon)**2 + y**2)**1.5
    D2 = ((x - earth)**2 + y**2)**1.5

    ax = (x + 2.0 * y_prime - earth * (x + moon) / D1 - moon * (x - earth) / D2)
    ay = (y - 2.0 * x_prime - earth * y / D1 - moon * y / D2)

    return np.array([0, 0, ax, ay], dtype=float)

operators = [drift, kick]
methods = {
    (0,): "", # Default
    (1,): "RK4",
    (2,): "RK4"
}

print("BEGINNING SOLVE!")
start_time = time.perf_counter()

result = fs.fractional_step(operators, dt, Y0, t0, tf, "Strang", methods, fname="results.csv")

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")