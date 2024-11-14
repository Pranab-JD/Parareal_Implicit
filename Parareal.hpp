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
    
    CN(A, u, tol, dt_coarse, iters, LHS_matrix, rhs_vector);
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

    for (int m = 0; m < num_fine_steps_per_coarse; m++) 
    {
        CN(A, u, tol, dt_fine, iters, LHS_matrix, rhs_vector);
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
    int iters_c, iters_f, iters_t;                  //* GMRes iterations counter for coars and fine solvers
    int N = u.size();                               //* Number of elements in vector 'u'
    int max_parareal_iters = num_coarse_steps;      //* Maximum number of parareal iterations

    //* Initi vectors
    std::vector<std::vector<Eigen::VectorXd>> u_coarse(max_parareal_iters, std::vector<Eigen::VectorXd>(num_coarse_steps+1, Eigen::VectorXd::Zero(N)));
    std::vector<std::vector<Eigen::VectorXd>> u_parareal(max_parareal_iters, std::vector<Eigen::VectorXd>(num_coarse_steps+1, Eigen::VectorXd::Zero(N)));
    std::vector<Eigen::VectorXd> u_fine(std::vector<Eigen::VectorXd>(num_coarse_steps, Eigen::VectorXd::Zero(N)));

    Eigen::VectorXd u_temp;
    std::vector<double> u_error(num_coarse_steps);

    for (int kk = 0; kk < max_parareal_iters; kk ++)
    {
        u_coarse[kk][0] = u;
    }

    LeXInt::timer init_coarse, fine_solver, new_coarse;

    //? Initial coarse solve (0th parareal iteration)
    init_coarse.start();
    for (int nn = 0; nn < num_coarse_steps; nn++)
    {
        u_temp = u_coarse[0][nn];
        coarse(A, u_temp, tol, iters_c, LHS_matrix, rhs_vector, T, num_coarse_steps);
        u_coarse[0][nn+1] = u_temp;
        iters = iters + iters_c;
    }
    init_coarse.stop();

    //* Set parareal solution to current coarse solution
    u_parareal = u_coarse;

    // int converged_till = 0;

    //? Parareal iterations
    for (int kk = 0; kk < max_parareal_iters-1; kk++)
    {
        std::cout << std::endl << "Parareal #: " << kk + 1 << std::endl;
        // std::cout << "Converged till: " << converged_till << std::endl;

        //? Fine solve (from previous parareal solution)
        fine_solver.start();
        #pragma omp parallel for
        for (int nn = 0; nn < num_coarse_steps; nn++)
        {
            Eigen::VectorXd u_local = u_parareal[kk][nn];
            Eigen::SparseMatrix<double> LHS_matrix_local = LHS_matrix;
            Eigen::VectorXd rhs_vector_local = rhs_vector;
            
            fine(A, u_local, tol, iters_f, LHS_matrix_local, rhs_vector_local, T, num_coarse_steps, num_fine_steps_per_coarse);
            
            #pragma omp critical
            {
                u_fine[nn] = u_local;
                iters = iters + iters_f;
            }
        }
        fine_solver.stop();

        //? New coarse solve
        new_coarse.start();
        for (int nn = 0; nn < num_coarse_steps; nn++)
        {
            u_temp = u_parareal[kk][nn];
            coarse(A, u_temp, tol, iters_c, LHS_matrix, rhs_vector, T, num_coarse_steps);
            u_coarse[kk+1][nn+1] = u_temp;
            iters = iters + iters_c;
        }
        new_coarse.stop();

        //* Update parareal solution
        for (int nn = 0; nn < num_coarse_steps; nn++)
        {
            u_parareal[kk+1][nn+1] = u_coarse[kk+1][nn] + u_fine[nn] - u_coarse[kk][nn];
        }

        //* Check convergence -- compute error estimate
        for (int nn = 0; nn < num_coarse_steps; nn++)
        {
            u_temp = u_parareal[kk+1][nn] - u_parareal[kk][nn];
            u_error[nn] = u_temp.norm()/u_parareal[kk+1][nn].norm();

            u = u_parareal[kk+1][nn+1];
            // if (u_error[nn] < 1e-6)
            //     converged_till = nn;
        }

        std::cout << "Error: " << std::endl;
        for (int nn = 0; nn < num_coarse_steps; nn++)
            std::cout << u_error[nn] << "      ";
        std::cout << std::endl;

        if (u_error[num_coarse_steps-1] < 1e-6) 
        {
            std::cout << "Convergence reached after " << kk + 1 << " iterations." << std::endl;
            break;
        }
    }
    
    std::cout << "Initial Coarse solve (s)  : " << init_coarse.total() << std::endl;
    std::cout << "Fine solve (s)            : " << fine_solver.total() << std::endl;
    std::cout << "New Coarse solve (s)      : " << new_coarse.total() << std::endl << std::endl;
}