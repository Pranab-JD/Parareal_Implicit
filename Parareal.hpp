#pragma once

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <iostream>
#include <iomanip>
#include <cmath>

#include "functions.hpp"
#include "Implicit.hpp"

//? Eigen
#include <Eigen/Sparse>
#include <Eigen/Dense>


//? Coarse solver
void coarse(const Eigen::SparseMatrix<double>& A,              //* Matrix A (input)
                  Eigen::VectorXd& u,                          //* Solution vector x (output)
                  double& tol,                                 //* Desired accuracy (input)
                  int& iters,                                  //* Number of iterations (output)
                  Eigen::SparseMatrix<double>& LHS_matrix,     //* LHS matrix (A in Ax = b) 
                  Eigen::VectorXd& rhs_vector,                 //* RHS vector (b in Ax = b) 
                  double T,                                    //* Total simulation time 
                  int num_coarse_steps                         //* Number of coarse time steps 
                  )   
{
    double dt_coarse = T/num_coarse_steps;
    
    for (int n = 0; n < num_coarse_steps; ++n) 
    {
        CN(A, u, tol, dt_coarse, iters, LHS_matrix, rhs_vector);
    }
}


//? Fine solver
void fine(const Eigen::SparseMatrix<double>& A,              //* Matrix A (input)
                Eigen::VectorXd& u,                          //* Solution vector x (output)
                double& tol,                                 //* Desired accuracy (input)
                int& iters,                                  //* Number of iterations (output)
                Eigen::SparseMatrix<double>& LHS_matrix,     //* LHS matrix (A in Ax = b) 
                Eigen::VectorXd& rhs_vector,                 //* RHS vector (b in Ax = b) 
                double T,                                    //* Total simulation time 
                int num_coarse_steps,                        //* Number of coarse time steps 
                int num_fine_steps_per_coarse                //* Number of fine steps per coarse step 
                ) 
{
    double dt_fine = T/(num_coarse_steps * num_fine_steps_per_coarse);
    
    // #pragma omp parallel for num_threads(num_coarse_steps)
    for (int n = 0; n < num_coarse_steps; ++n) 
    {
        // Eigen::SparseMatrix<double> LHS_matrix_local = LHS_matrix;
        // Eigen::VectorXd rhs_vector_local = rhs_vector;
        // Eigen::VectorXd u_local = u;
        
        for (int m = 0; m < num_fine_steps_per_coarse; ++m) 
        {
            CN(A, u_local, tol, dt_fine, iters, LHS_matrix_local, rhs_vector_local);
        }
    }
}


//? Parareal algorithm
void parareal(const Eigen::SparseMatrix<double>& A,        //* Matrix A (input)
              Eigen::VectorXd& u,                          //* Solution vector x (output)
              double& tol,                                 //* Desired accuracy (input)
              int& iters,                                  //* Number of iterations (output)
              Eigen::SparseMatrix<double>& LHS_matrix,     //* LHS matrix (A in Ax = b) 
              Eigen::VectorXd& rhs_vector,                 //* RHS vector (b in Ax = b) 
              double T,                                    //* Total simulation time 
              int num_coarse_steps,                        //* Number of coarse time steps 
              int num_fine_steps_per_coarse                //* Number of fine steps per coarse step  
              ) 
{
    //* Copy solution vectors
    Eigen::VectorXd u_coarse         = u;
    Eigen::VectorXd u_fine           = u;
    Eigen::VectorXd u_parareal       = u;
    Eigen::VectorXd u_parareal_prev  = u;

    LeXInt::timer init_coarse, fine_solver, parareal_iters;

    int iters_c, iters_f;                           //* GMRes iterations counter for coars and fine solvers
    int N = u.size();                               //* Number of elements in vector 'u'
    int max_parareal_iters = num_coarse_steps;      //* Maximum number of parareal iterations

    //? Initial coarse solve
    init_coarse.start();
    coarse(A, u_coarse, tol, iters_c, LHS_matrix, rhs_vector, T, num_coarse_steps);
    init_coarse.stop();

    //* Set parareal solution to current coarse solution
    u_parareal = u_coarse;

    //? Parareal iterations
    parareal_iters.start();
    for (int k = 0; k < max_parareal_iters; ++k) 
    {
        //? Fine solve (from previous parareal solution)
        fine_solver.start();
        fine(A, u_fine, tol, iters_f, LHS_matrix, rhs_vector, T, num_coarse_steps, num_fine_steps_per_coarse);
        fine_solver.stop();

        //? New coarse solve
        Eigen::VectorXd u_coarse_new = u_parareal;
        coarse(A, u_coarse_new, tol, iters_c, LHS_matrix, rhs_vector, T, num_coarse_steps);

        //* Store current parareal solution for convergence check
        u_parareal_prev = u_parareal;

        //* Update parareal solution
        u_parareal = u_coarse_new + u_fine - u_coarse;

        //* Update the coarse solution for the next parareal iteration
        u_coarse = u_coarse_new;

        //* Compute error estimate
        Eigen::VectorXd u_error = u_parareal - u_parareal_prev;
        double error = u_error.norm()/N;
        std::cout << "Iteration " << k + 1 << ", error: " << error << std::endl;

        if (error < 1e-8) 
        {
            std::cout << "Convergence reached after " << k + 1 << " iterations." << std::endl;
            break;
        }
    }
    parareal_iters.stop();

    cout << "Total time elapsed for parareal (s) : " << init_coarse.total() + parareal_iters.total() << endl;
    cout << "          Initial Coarse solver (s) : " << init_coarse.total() << endl;
    cout << "                    Fine solver (s) : " << fine_solver.total() << endl;
    cout << "            Parareal iterations (s) : " << parareal_iters.total() << endl;
}