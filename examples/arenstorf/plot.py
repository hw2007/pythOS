import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("results.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

# Plot it!
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Arenstorf Orbit")
plt.axis("equal")

plt.savefig("graph.png")