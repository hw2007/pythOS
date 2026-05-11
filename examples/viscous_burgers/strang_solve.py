# Equation: viscous Burgers, u_t = ν u_xx - 1/2 * (u^2)_x

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

# Viscosity coefficient ν
VISC = Constant(0.0005)

dt = 1/10
tf = 5 # Initial time is locked at zero, this defines final time
num_steps = int(tf / dt)

# Begin with random noise
np.random.seed(314)
u.dat.data[:] = 0.1 * np.random.randn(len(u.dat.data))

boundary_condition = DirichletBC(V, 0, "on_boundary")

# Viscosity term
a_viscosity = trial_u * v * dx + dt * VISC * dot(grad(trial_u), grad(v)) * dx
L_viscosity = u_prev * v * dx

viscosity_problem = LinearVariationalProblem(a_viscosity, L_viscosity, u, bcs=boundary_condition)
viscosity_solver = LinearVariationalSolver(viscosity_problem)

# Convective term
F = (trial_u - u_prev)/(dt/2) * v * dx - 0.5 * trial_u**2 * v.dx(0) * dx # We use dt/2, because this is the term that is applied half twice (Strang splitting)

convective_problem = NonlinearVariationalProblem(F, u, bcs=boundary_condition)
convective_solver = NonlinearVariationalSolver(convective_problem)


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
    u_prev.assign(u) # Make prev state the same as current state

    # Convection half-step
    convective_solver.solve()

    # Apply viscosity full step
    viscosity_solver.solve() # Perform the function

    # Convection half-step
    convective_solver.solve()

    print(f"Stepped {step}/{num_steps} steps")

    if step % (num_steps // 5) == 0: snapshot(step)

print("DONE! Time to plot...")

# PLOTTING

snapshot(num_steps)