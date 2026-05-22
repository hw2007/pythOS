import butcher_tableau as bt
import numpy as np

# Using classic runge-kutta method
# Created by Martin Kutta (1901) in "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen"

# RK4

c4 = np.array([0, 1/2, 1/2, 1])

A4 = np.array([
    [0, 0, 0, 0],
    [1/2, 0, 0, 0],
    [0, 1/2, 0, 0],
    [0, 0, 1, 0]
])

b4 = np.array([1/6, 1/3, 1/3, 1/6])

rk4 = bt.Tableau(c=c4, a=A4, b=b4)

# RK3

c3 = np.array([0, 1/2, 1])

A3 = np.array([
    [0, 0, 0],
    [1/2, 0, 0],
    [-1, 2, 0]
])

b3 = np.array([1/6, 2/3, 1/6])

rk3 = bt.Tableau(c=c3, a=A3, b=b3)

# Bogacki-Shampine Order 3(2) method

c_bs = np.array([0, 1/2, 3/4, 1])

A_bs = np.array([
    [0, 0, 0, 0],
    [1/2, 0, 0, 0],
    [0, 3/4, 0, 0],
    [2/9, 1/3, 4/9, 0]
])

b_bs = np.array([2/9, 1/3, 4/9, 0])

b_aux_bs = np.array([7/24, 1/4, 1/3, 1/8])

bogacki_shampine = bt.EmbeddedTableau(c=c_bs, a=A_bs, b=b_bs, b_aux=b_aux_bs, order=2)

# Heun-Euler Order 2(1) method

c_he = np.array([0, 1])

A_he = np.array([
    [0, 0],
    [1, 0]
])

b_he = np.array([1/2, 1/2])

b_aux_he = np.array([1, 0])

heun_euler = bt.EmbeddedTableau(c=c_he, a=A_he, b=b_he, b_aux=b_aux_he, order=1)