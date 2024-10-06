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

void CN()
{
    // TODO: Implement CN (the function declaration & parameters have to be similar to the previous one)
}
