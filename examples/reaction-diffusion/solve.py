# Equation: reaction-diffusion, u_t = D delta u + f(u)

from firedrake import *
import fractional_step
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

SIZE = 100

mesh = UnitSquareMesh(SIZE, SIZE) # Create spatial mesh
V = FunctionSpace(mesh, "CG", 1) # "CG" stands for continuous Galerkin

# Create a function that exists in finite space V. Will store the u values from the equation above
u = Function(V)
u_prev = Function(V) # Solution at previous timestep

# Used in the weak form
v = TestFunction(V)

u_prev = Function(V) # u at previous timstep

D = Constant(0.0005)

dt = 1/10
tf = 10 # Initial time is locked at zero, this defines final time
num_steps = int(tf // dt)

# Define coordinates in mesh
x, y = SpatialCoordinate(mesh)

# Initial condition
#u0 = exp(-SIZE * ((x - 0.5)**2 + (y - 0.5)**2)) # Creates a smooth "bump" in the centre of the mesh
# Load initial condition into the space
#u.interpolate(u0)

# Begin with random noise
np.random.seed(314)
u.dat.data[:] = 0.1 * np.random.randn(len(u.dat.data))

# function for f(u).
def fu(u):
    return u - u**3

# Weak form of the equation. I am not totally sure what this means yet, was taken from online.
weak = (
    (u - u_prev) / dt * v * dx
    + D * dot(grad(u), grad(v)) * dx
    - fu(u) * v * dx
)

boundary_condition = DirichletBC(V, 0, "on_boundary")

problem = NonlinearVariationalProblem(weak, u, bcs=boundary_condition)
solver = NonlinearVariationalSolver(problem)

# Solve !!!
for step in range(num_steps):
    u_prev.assign(u) # Make prev state the same as current state
    solver.solve() # Perform the function
    print(f"Stepped {step}/{num_steps} steps")

print("DONE! Time to plot...")

# PLOTTING

coords = mesh.coordinates.dat.data_ro
cells = mesh.coordinates.cell_node_map().values.reshape(-1, 3)

triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
plt.figure()
plt.tripcolor(triang, u.dat.data_ro, shading="gouraud")
plt.colorbar(label="u value")
plt.gca().set_aspect("equal")
plt.title(f"Discrete Space at t = {tf}")
plt.savefig("graph.png")