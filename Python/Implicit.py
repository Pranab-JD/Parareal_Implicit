"""
Created on Thu Nov 14 22:18 2024

@author: Pranab JD

"""

from scipy.sparse import identity, linalg

class counter:
    def __init__(self):
        self.count = 0
    def incr(self, x):
        self.count = self.count + 1

def GMRES(A, b, x0, tol):
    c = counter()
    return linalg.gmres(A, b, x0 = x0, callback = c.incr, tol = tol), c.count

def CG(A, b, x0, tol):
    c = counter()
    return linalg.cg(A, b, x0 = x0, callback = c.incr, tol = tol), c.count

def Crank_Nicolson(u, dt, A, tol):
    u, iters = GMRES(identity(A.shape[0]) - 0.5*dt*A, u + 0.5*dt*A.dot(u), u, tol)
    return u[0], iters + 1