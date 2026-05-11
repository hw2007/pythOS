# Equation: reaction-diffusion, u_t = D delta u + f(u)

from firedrake import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

SIZE = 100

mesh = UnitSquareMesh(SIZE, SIZE) # Create spatial mesh
V = FunctionSpace(mesh, "CG", 1) # "CG" stands for continuous Galerkin

# Create a function that exists in finite space V. Will store the u values from the equation above
u = Function(V)
u_prev = Function(V) # Solution at previous timestep

# Used in solving laplacian for diffusion
v = TestFunction(V)
trial_u = TrialFunction(V)

u_prev = Function(V) # u at previous timstep

D = Constant(0.0005)

dt = 1/10
tf = 5 # Initial time is locked at zero, this defines final time
num_steps = int(tf / dt)

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
def reaction(dt, u):
    u.interpolate(u + dt * (u - u**3))

a_diffusion = trial_u * v * dx + dt * D * dot(grad(trial_u), grad(v)) * dx
L_diffusion = u_prev * v * dx

boundary_condition = DirichletBC(V, 0, "on_boundary")
diffusion_problem = LinearVariationalProblem(a_diffusion, L_diffusion, u, bcs=boundary_condition)
diffusion_solver = LinearVariationalSolver(diffusion_problem)

def snapshot(i):
    coords = mesh.coordinates.dat.data_ro
    cells = mesh.coordinates.cell_node_map().values.reshape(-1, 3)

    triang = tri.Triangulation(coords[:, 0], coords[:, 1], cells)
    plt.figure()
    plt.tripcolor(triang, u.dat.data_ro, shading="gouraud")
    plt.colorbar(label="u value")
    plt.gca().set_aspect("equal")
    plt.title(f"Discrete Space at t = {tf / num_steps * i}")
    plt.savefig(f"graph{i}.png")

snapshot(0)

# Solve !!!
for step in range(num_steps):
    # Reaction half-step
    reaction(dt/2, u)

    # Diffusion full step
    u_prev.assign(u) # Make prev state the same as current state
    diffusion_solver.solve() # Perform the function

    # Reaction half-step
    reaction(dt/2, u)

    print(f"Stepped {step}/{num_steps} steps")

    if step % (num_steps // 5) == 0: snapshot(step)

print("DONE! Time to plot...")

# PLOTTING

snapshot(num_steps)