import butcher_tableau as bt
import numpy as np

# HEUN'S METHOD W/ EMBEDDED FORWARD EULER (EXPLICIT)

c_heun = np.array([0, 1])

A_heun = np.array([
    [0, 0],
    [1, 0]
])

b_heun = np.array([1/2, 1/2])

b_fe = np.array([1, 0])

heun_fe = bt.EmbeddedTableau(c=c_heun, a=A_heun, b=b_heun, b_aux=b_fe, order=1)

# SDIRK2 W/ EMBEDDED BACLWARD EULER (IMPLICIT)

GAMMA = (2 - np.sqrt(2)) / 2

c_sd2 = np.array([GAMMA, 1])

A_sd2 = np.array([
    [GAMMA, 0],
    [1 - GAMMA, GAMMA]
])

b_sd2 = np.array([1 - GAMMA, GAMMA])

b_be = np.array([1, 0])

sd2_be = bt.EmbeddedTableau(c=c_sd2, a=A_sd2, b=b_sd2, b_aux=b_be, order=1)

# 2(1) Embedded pair from Portero 2012 paper

c = np.array([1, 1, 4/9, 1/3])

A = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [-343/180, 5/9, 47/20, 0],
    [-1592/300, -121/60, 564/100, 47/20]
])

b_main = [1/10, 1/4, 9/10, 3/4]
b_embedded = [1, 1, 0, 0]

portero_2_1 = bt.EmbeddedTableau(c=c, a=A, b=b_main, b_aux=b_embedded, order=1)