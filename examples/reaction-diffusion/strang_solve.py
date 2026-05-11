# Equation: reaction-diffusion, u_t = D delta u + f(u)

from firedrake import *
import fractional_step as fs
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import time
import h5py

SIZE = 100

mesh = UnitSquareMesh(SIZE, SIZE) # Create spatial mesh
V = FunctionSpace(mesh, "CG", 1) # "CG" stands for continuous Galerkin

D = Constant(0.0005)

dt = 1/10
t0 = Constant(0)
tf = 2

# Define coordinates in mesh
x, y = SpatialCoordinate(mesh)

# Initial condition
#u0 = exp(-SIZE * ((x - 0.5)**2 + (y - 0.5)**2)) # Creates a smooth "bump" in the centre of the mesh
# Load initial condition into the space
#u.interpolate(u0)

# Begin with random noise
np.random.seed(314)
u0 = Function(V)
u0.dat.data[:] = 0.1 * np.random.randn(len(u0.dat.data))

def diffusion(t, u):
    f = Function(V)
    v = TestFunction(V)
    trial = TrialFunction(V)

    a = trial * v * dx
    L = -D * dot(grad(u), grad(v)) * dx

    solve(a == L, f)
    
    return f

# function for f(u).
def reaction(t, u):
    f = Function(V)
    f.interpolate(u - u**3)
    return f

operators = [diffusion, reaction]
methods = {
    (0,): "", # Default
    (1,): "RK3",
    (2,): "RK3"
}

print("BEGINNING SOLVE!")
start_time = time.perf_counter()

result = fs.fractional_step(operators, dt, u0, t0, tf, "Strang", methods, fname="results.h5")

end_time = time.perf_counter()
print(f"DONE! Solved in {end_time - start_time} seconds.")
print("Time to plot...")

# PLOTTING
def print_structure(name, obj):
    print(name)

with h5py.File("results.h5", "r") as f:
    print(list(f["times/idx"].keys()))
with CheckpointFile("results.h5", "r") as afile:
    loaded_mesh = afile.load_mesh()

coords = loaded_mesh.coordinates.dat.data_ro
cells = loaded_mesh.coordinates.cell_node_map().values.reshape(-1, 3)

triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
plt.figure()
plt.tripcolor(triang, u.dat.data_ro, shading="gouraud")
plt.colorbar(label="u value")
plt.gca().set_aspect("equal")
plt.title(f"Discrete Space at t = {tf}")
plt.savefig("graph.png")