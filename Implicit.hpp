#pragma once

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>

#include "functions.hpp"

//? Eigen
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

//? ----------------------------------------------------------
//?
//? Description:
//?     A collection of implicit integrators and solvers
//?
//? ----------------------------------------------------------

void GMRes(const Eigen::SparseMatrix<double>& A,        //* Matrix A (input)
           const Eigen::VectorXd& b,                    //* RHS vector b (input)
           Eigen::VectorXd& x,                          //* Solution vector x (output)
           double tol,                                  //* Desired accuracy (input)
           int& iters                                   //* Number of iterations (output)
           ) 
{
    //? Initialise solver
    Eigen::GMRES<Eigen::SparseMatrix<double> > solver(A);
    solver.setTolerance(tol);

    //? Compute solution
    x = solver.solve(b);

    //? Number of ierations needed to converge
    iters = solver.iterations();
}

void implicit_Euler(const Eigen::SparseMatrix<double>& A,        //* Matrix A (input)
                    Eigen::VectorXd& u,                          //* Solution vector x (output)
                    double& tol,                                 //* Desired accuracy (input)
                    double& dt,                                  //* Time step size
                    int& iters,                                  //* Number of iterations (output)
                    Eigen::SparseMatrix<double>& LHS_matrix,     //* LHS matrix (A in Ax = b) 
                    Eigen::VectorXd& rhs_vector                  //* RHS vector (b in Ax = b) 
                    )
{
    //? Create an identity matrix of same size as A
    Eigen::SparseMatrix<double> I(A.rows(), A.rows());
    I.setIdentity();

    //* LHS matrix: (I - dt*A)
    LHS_matrix = I - dt*A;

    //* RHS vector: u
    rhs_vector = u;

    //? Solve using GMRes (Ax = b :: LHS_matrix * u = rhs_vector)
    GMRes(LHS_matrix, rhs_vector, u, tol, iters);
}

void CN(const Eigen::SparseMatrix<double>& A,              //* Matrix A (input)
              Eigen::VectorXd& u,                          //* Solution vector x (output)
        const double& tol,                                 //* Desired accuracy (input)
        const double& dt,                                  //* Time step size
              int& iters,                                  //* Number of iterations (output)
              Eigen::SparseMatrix<double>& LHS_matrix,     //* LHS matrix (A in Ax = b) 
              Eigen::VectorXd& rhs_vector                  //* RHS vector (b in Ax = b) 
              )
{
    //? Create an identity matrix of same size as A
    Eigen::SparseMatrix<double> I(A.rows(), A.rows());
    I.setIdentity();

    //* LHS matrix: (I - dt*A)
    LHS_matrix = I - 0.5*dt*A;

    //* RHS vector: u
    rhs_vector = u + 0.5*dt*(A*u);

    //? Solve using GMRes (Ax = b :: LHS_matrix * u = rhs_vector)
    GMRes(LHS_matrix, rhs_vector, u, tol, iters);
}

void RK2(const Eigen::SparseMatrix<double>& A,
               Eigen::VectorXd& u,
         const double& dt)
{
    //? Internal stage 1: k1 = dt * RHS(u)
    Eigen::VectorXd k1 = dt*A*u;                    //* k1 = RHS(u) = du/dt

    //? Internal stage 2: k2 = dt * RHS(u + k1)
    Eigen::VectorXd u_temp = u + k1;
    Eigen::VectorXd k2 = dt*A*u_temp;               //* k2 = RHS(u + k1) = RHS(u_sol) 

    //? u_rk2 = u + 1./2.*(k1 + k2)
    u = u + 0.5*(k1 + k2);
}