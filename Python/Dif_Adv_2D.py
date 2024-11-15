"""
Created on Fri Nov 15 00:16 2024

@author: Pranab JD

Description: Create required matrices and initialize parameters
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, kron, identity, diags

from Implicit import *
from Parareal import *

### ============================================================================ ###

N = 2**7
X = np.linspace(-1, 1, N, endpoint=False)
Y = np.linspace(-1, 1, N, endpoint=False)
dx = X[2] - X[1]
dy = Y[2] - Y[1]

X, Y = np.meshgrid(X, Y)

velocity = 0.2

###? 2nd order centered difference for diffusion (1, -2, 1)
Dif_x = lil_matrix(diags(np.ones(N - 1), -1) + diags(-2*np.ones(N), 0) + diags(np.ones(N - 1), 1))
Dif_y = lil_matrix(diags(np.ones(N - 1), -1) + diags(-2*np.ones(N), 0) + diags(np.ones(N - 1), 1))
Dif_x[0, -1] = 1; Dif_x[-1, 0] = 1 
Dif_y[0, -1] = 1; Dif_y[-1, 0] = 1

A_dif = kron(identity(N), Dif_x/dx**2) + kron(Dif_y/dy**2, identity(N))

###? 3rd order upwind for advection (-2/6, -3/6, 1, -1/6)
Adv_x = lil_matrix(diags(-2/6*np.ones(N - 1), -1) + diags(-3/6*np.ones(N), 0) + diags(np.ones(N - 1), 1) + diags(-1/6*np.ones(N - 2), 2))
Adv_y = lil_matrix(diags(-2/6*np.ones(N - 1), -1) + diags(-3/6*np.ones(N), 0) + diags(np.ones(N - 1), 1) + diags(-1/6*np.ones(N - 2), 2))
Adv_x[-2, 0] = -1/6; Adv_x[-1, 0] = 1; Adv_x[-1, 1] = -1/6; Adv_x[0, -1] = -2/6
Adv_y[-2, 0] = -1/6; Adv_y[-1, 0] = 1; Adv_y[-1, 1] = -1/6; Adv_y[0, -1] = -2/6

A_adv = kron(identity(N), Adv_x*velocity/dx) + kron(Adv_y*velocity/dy, identity(N))

A = A_dif + A_adv

### -------------------------------------- ###

###* CFL
dif_cfl = (dx**2 * dy**2)/(2*(dx**2 + dy**2))
adv_cfl = min(dx/abs(velocity), dy/abs(velocity))

###* Temporal parameters
N_cfl = 2.0
dt = N_cfl * min(dif_cfl, adv_cfl)
num_time_steps = 1000        # Simulation time
T = num_time_steps * dt
tol = 1e-8

###* Parareal parameters
num_coarse_steps = 10
num_fine_steps_per_coarse = 4

### -------------------------------------- ###

u_init = 1 + 10 * np.exp(-(((X + 0.5) * (X + 0.5)) + ((Y + 0.5) * (Y + 0.5)) )/ 0.02)

u_serial = u_init.reshape(N*N)
u_parareal = u_init.reshape(N*N)

if __name__ == "__main__":

    ###? Parareal algorithm
    u_parareal = parareal(T, u_parareal, A, num_coarse_steps, num_fine_steps_per_coarse, tol)

    ###? Serial solution
    for nn in range(num_time_steps):
        u_serial, _ = Crank_Nicolson(u_serial, dt, A, tol)

    print()
    print("Error wrt to serial: ", np.linalg.norm(u_serial - u_parareal[num_coarse_steps, :])/np.linalg.norm(u_serial))

    plt.figure(figsize = (12, 6), dpi = 100)
    plt.subplot(1, 2, 1)
    plt.imshow(u_serial.reshape(N, N), origin = "lower", cmap = cm.plasma, extent = [-1, 1, -1, 1], aspect = 'equal')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(u_parareal[num_coarse_steps-1, :].reshape(N, N), origin = "lower", cmap = cm.plasma, extent = [-1, 1, -1, 1], aspect = 'equal')
    plt.colorbar()
    plt.show()
