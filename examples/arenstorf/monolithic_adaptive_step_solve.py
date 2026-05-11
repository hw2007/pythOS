from scipy.integrate import solve_ivp
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
tf = 17.06521656015796 * 1

def arenstorf(t, Y):
    x, y, x_prime, y_prime = Y

    D1 = ((x + moon)**2 + y**2)**1.5
    D2 = ((x - earth)**2 + y**2)**1.5

    ax = (x + 2.0 * y_prime - earth * (x + moon) / D1 - moon * (x - earth) / D2)
    ay = (y - 2.0 * x_prime - earth * y / D1 - moon * y / D2)

    return np.array([x_prime, y_prime, ax, ay], dtype=float)

print("BEGINNING SOLVE!")
start_time = time.perf_counter()

# rtol is relative tolerance. Relative to size of values. If values are larger, a larger error is allowed.
# atol is absolute error. Comes into effect with values near zero. If we only used rtol, near-0 values would need near infinite precision. (bad)
result = solve_ivp(arenstorf, (t0, tf), Y0, method="RK45", rtol=1e-9, atol=1e-12)

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")

# Save to file
t = result.t
Y = result.y.T

with open("results.csv", "w") as f:
    for i in range(len(t)):
        f.write(f"{t[i]},{Y[i,0]},{Y[i,1]},{Y[i,2]},{Y[i,3]}\n")