import butcher_tableau as bt
import numpy as np

# Using classic runge-kutta method
# Created by Martin Kutta (1901) in "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen"

c = np.array([0, 1/2, 1/2, 1])

A = np.array([
    [0, 0, 0, 0],
    [1/2, 0, 0, 0],
    [0, 1/2, 0, 0],
    [0, 0, 1, 0]
])

b = np.array([1/6, 1/3, 1/3, 1/6])

rk4_method = bt.Tableau(c=c, a=A, b=b)