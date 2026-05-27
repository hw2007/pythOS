"""
Compute f(t,x,y) from the PDE
"""
 
import sympy as sp
 
# Define symbols
t, x, y = sp.symbols('t x y', real=True)
 
# Define the exact solution
u = 3 * t * sp.exp(-3*t + 1) * sp.sin(sp.pi * x**2) * (sp.sin(sp.pi * y)**2)
 
print("EXACT SOLUTION:")
print("u(t,x,y) =")
sp.pprint(u)
 
# Compute all needed partial derivatives
print("PARTIAL DERIVATIVES")
 
u_t = sp.diff(u, t)
print("\nu_t =")
sp.pprint(u_t)
 
u_xx = sp.diff(u, x, 2)
print("\nt_xx =")
sp.pprint(u_xx)
 
u_yy = sp.diff(u, y, 2)
print("\nu_yy =")
sp.pprint(u_yy)
 
# Compute Laplacian
lap_u = u_xx + u_yy
print("\n∆u = u_xx + u_yy =")
sp.pprint(lap_u)
 
# Compute f from the PDE
print("COMPUTING f")
 
f = u_t - (1 + sp.exp(-t)) * x * y * lap_u + u
 
print("\nRaw form (before simplification):")
sp.pprint(f)
 
# Simplify
print("\nSimplifying...")
f_simplified = sp.simplify(f)
 
print("\nSimplified form:")
sp.pprint(f_simplified)
 
# Factor
print("\nFactored form:")
f_factored = sp.factor(f_simplified)
sp.pprint(f_factored)
 
# Verification: check that the PDE is satisfied
print("VERIFICATION")
print("Checking things zero out when moving all to RHS")
 
X = u_t - (1 + sp.exp(-t)) * x * y * lap_u + u - f
X = sp.simplify(X)
 
print("X (should be 0):")
sp.pprint(X)
