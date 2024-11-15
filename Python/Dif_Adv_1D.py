"""
Created on Thu Nov 14 22:19 2024

@author: Pranab JD

Description: Create required matrices and initialize parameters
"""

import numpy as np
from scipy.sparse import lil_matrix, kron, identity, diags

from Implicit import *
from Parareal import *

import matplotlib.pyplot as plt

### ============================================================================ ###

N_x = 2**12
X = np.linspace(-1, 1, N_x, endpoint=False)
dx = X[2] - X[1]

velocity = 0.2

###? 2nd order centered difference for diffusion (1, -2, 1) & (-1/2, 0, 1/2)
A_dif = lil_matrix(diags(np.ones(N_x - 1), -1) + diags(-2*np.ones(N_x), 0) + diags(np.ones(N_x - 1), 1))
A_dif[0, -1] = 1; A_dif[-1, 0] = 1 
A_dif = A_dif/dx**2

###? 3rd order upwind for advection (-2/6, -3/6, 1, -1/6)
A_adv = lil_matrix(diags(-2/6*np.ones(N_x - 1), -1) + diags(-3/6*np.ones(N_x), 0) + diags(np.ones(N_x - 1), 1) + diags(-1/6*np.ones(N_x - 2), 2))
A_adv[-2, 0] = -1/6; A_adv[-1, 0] = 1; A_adv[-1, 1] = -1/6; A_adv[0, -1] = -2/6
A_adv = A_adv*velocity/dx

A = A_dif + A_adv

### -------------------------------------- ###

###* CFL
dif_cfl = dx**2
adv_cfl = dx/abs(velocity)

###* Temporal parameters
N_cfl = 0.5
dt = N_cfl * min(dif_cfl, adv_cfl)
num_time_steps = 5000        # Simulation time
T = num_time_steps * dt
tol = 1e-12

###* Parareal parameters
num_coarse_steps = 5
num_fine_steps_per_coarse = 5

### -------------------------------------- ###

u_init = 1 + 10 * np.exp(-((X + 0.5) * (X + 0.5)) / 0.02)

u_serial = u_init
u_parareal = u_init

if __name__ == "__main__":

    ###? Parareal algorithm
    u_parareal = parareal(T, u_parareal, A, num_coarse_steps, num_fine_steps_per_coarse, tol)

    ###? Serial solution
    for nn in range(num_time_steps):
        u_serial, _ = Crank_Nicolson(u_serial, dt, A, tol)

    print()
    print("Error wrt to serial: ", np.linalg.norm(u_serial - u_parareal[num_coarse_steps, :])/np.linalg.norm(u_serial))

    plt.plot(X, u_init, "g")
    plt.plot(X, u_serial, "bd")
    plt.plot(X, u_parareal[num_coarse_steps, :], "r.")
    plt.show()

    