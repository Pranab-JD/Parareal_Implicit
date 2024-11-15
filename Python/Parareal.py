"""
Created on Thu Nov 14 22:18 2024

@author: Pranab JD

"""

import numpy as np
import time

from Implicit import *

###! Fine solver: Solve with small time steps
def fine_solver(u, dt_fine, A, num_fine_steps, tol):
    for _ in range(num_fine_steps):
        u, iters = Crank_Nicolson(u, dt_fine, A, tol)
    return u

###! Coarse solver: Solve with large time steps
def coarse_solver(u, dt_coarse, A, tol):
    u, iters = Crank_Nicolson(u, dt_coarse, A, tol)
    return u

###! Parareal algorithm with Crank-Nicolson and GMRES
def parareal(T, u_init, A, num_coarse_steps, num_fine_steps_per_coarse, tol):

    max_parareal_iters = num_coarse_steps

    ##* Time step sizes
    dt_coarse = T/num_coarse_steps
    dt_fine = T/(num_coarse_steps * num_fine_steps_per_coarse)

    ##* Initializing vectors
    u_coarse = np.zeros((max_parareal_iters+2, num_coarse_steps+1, np.size(u_init)))
    u_fine = np.zeros((num_coarse_steps, np.size(u_init)))
    u_parareal = np.zeros((max_parareal_iters+2, num_coarse_steps+1, np.size(u_init)))
    u_error = np.zeros(num_coarse_steps+1)

    u_coarse[:, 0, :] = u_init

    ###? Initial coarse solve
    for nn in range(num_coarse_steps):
        u_coarse[0, nn+1, :] = coarse_solver(u_coarse[0, nn, :], dt_coarse, A, tol)

    u_parareal = u_coarse

    ###? Parareal iterations
    for k in range(max_parareal_iters + 1):
        
        print("Parareal #: ", k+1)

        ##* Fine solver step: Parallel computation of fine solver
        for nn in range(num_coarse_steps):
            u_fine[nn, :] = fine_solver(u_parareal[k, nn, :], dt_fine, A, num_fine_steps_per_coarse, tol)

        ##* New coarse solve (u_parareal[k, nn, :] or u_fine[nn, :])
        for nn in range(num_coarse_steps):
            u_coarse[k+1, nn+1, :] = coarse_solver(u_parareal[k, nn, :], dt_coarse, A, tol)

        ##* Update parareal solution
        for nn in range(num_coarse_steps):
            u_parareal[k+1, nn+1, :] = u_coarse[k+1, nn, :] + u_fine[nn, :] - u_coarse[k, nn, :]

        ##* Check for convergence (error between successive parareal iterations)
        for nn in range(0, num_coarse_steps+1):
            u_error[nn] = np.linalg.norm(u_parareal[k+1, nn, :] - u_parareal[k, nn, :])/np.linalg.norm(u_parareal[k+1, nn, :])

        if np.all(u_error < 1e-6):
            print("Error: ", u_error)
            print(f"Convergence reached after {k+1} iterations.")
            break

    return u_parareal[k, :, :]
