#pragma once

#include "Problems.hpp"
// #include "error_check.hpp"

//? Eigen
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;

//? ====================================================================================== ?//

struct RHS_Dif_Adv_2D:public Problems_2D
{
    //? RHS = A_adv.u^2/2.0 + A_dif.u

    //! Constructor
    RHS_Dif_Adv_2D(int _N, double _dx, double _dy, double _velocity) : Problems_2D(_N, _dx, _dy, _velocity) {}

    void Diffusion_matrix(Eigen::SparseMatrix<double>& Dif_xx, Eigen::SparseMatrix<double>& Dif_yy, Eigen::SparseMatrix<double>& I_N, Eigen::SparseMatrix<double>& A_dif)
    {
        vector<Eigen::Triplet<double>> D_x;  // To store non-zero values
        vector<Eigen::Triplet<double>> D_y;  // To store non-zero values

        for (int i = 0; i < N; ++i) 
        {
            //* (ii, jj)
            D_x.push_back(Eigen::Triplet<double>(i, i, -2.0));
            D_y.push_back(Eigen::Triplet<double>(i, i, -2.0));

            //* (ii, jj - 1)
            if (i > 0) 
            {
                D_x.push_back(Eigen::Triplet<double>(i, i - 1, 1.0));
                D_y.push_back(Eigen::Triplet<double>(i, i - 1, 1.0));
            }

            //* (ii, jj + 1)
            if (i < N - 1) 
            {
                D_x.push_back(Eigen::Triplet<double>(i, i + 1, 1.0));
                D_y.push_back(Eigen::Triplet<double>(i, i + 1, 1.0));
            }
        }

        //? PBC
        D_x.push_back(Eigen::Triplet<double>(0, N - 1, 1.0));
        D_x.push_back(Eigen::Triplet<double>(N - 1, 0, 1.0));

        D_y.push_back(Eigen::Triplet<double>(0, N - 1, 1.0));
        D_y.push_back(Eigen::Triplet<double>(N - 1, 0, 1.0));

        //* Set the values into the sparse matrices
        Dif_xx.setFromTriplets(D_x.begin(), D_x.end());
        Dif_yy.setFromTriplets(D_y.begin(), D_y.end());

        //? Compute Kronecker product
        A_dif = Eigen::kroneckerProduct(I_N, Dif_xx/(dx*dx)).eval() + Eigen::kroneckerProduct(Dif_yy/(dy*dy), I_N).eval();
    }

    void Advection_matrix(Eigen::SparseMatrix<double>& Adv_x, Eigen::SparseMatrix<double>& Adv_y, Eigen::SparseMatrix<double>& I_N, Eigen::SparseMatrix<double>& A_adv)
    {
        vector<Eigen::Triplet<double>> A_x;  // To store non-zero values
        vector<Eigen::Triplet<double>> A_y;  // To store non-zero values

        for (int i = 0; i < N; ++i) 
        {
            //* (ii, jj)
            A_x.push_back(Eigen::Triplet<double>(i, i, -3.0));
            A_y.push_back(Eigen::Triplet<double>(i, i, -3.0));

            //* (ii, jj - 1)
            if (i > 0) 
            {
                A_x.push_back(Eigen::Triplet<double>(i, i - 1, -2.0));
                A_y.push_back(Eigen::Triplet<double>(i, i - 1, -2.0));
            }

            //* (ii, jj + 1)
            if (i < N - 1)
            {
                A_x.push_back(Eigen::Triplet<double>(i, i + 1, 1.0));
                A_y.push_back(Eigen::Triplet<double>(i, i + 1, 1.0));
            }

            //* (ii, jj + 2)
            if (i < N - 2) 
            {
                A_x.push_back(Eigen::Triplet<double>(i, i + 2, -1.0));
                A_y.push_back(Eigen::Triplet<double>(i, i + 2, -1.0));
            }
        }

        //? PBC
        A_x.push_back(Eigen::Triplet<double>(N - 2, 0, -1.0));
        A_x.push_back(Eigen::Triplet<double>(N - 1, 0,  1.0));
        A_x.push_back(Eigen::Triplet<double>(N - 1, 1, -1.0));
        A_x.push_back(Eigen::Triplet<double>(0, N - 1, -2.0));

        A_y.push_back(Eigen::Triplet<double>(N - 2, 0, -1.0));
        A_y.push_back(Eigen::Triplet<double>(N - 1, 0,  1.0));
        A_y.push_back(Eigen::Triplet<double>(N - 1, 1, -1.0));
        A_y.push_back(Eigen::Triplet<double>(0, N - 1, -2.0));
        
        //* Set the values into the sparse matrices
        Adv_x.setFromTriplets(A_x.begin(), A_x.end());
        Adv_y.setFromTriplets(A_y.begin(), A_y.end());

        //? Compute Kronecker product
        A_adv = Eigen::kroneckerProduct(I_N, velocity*Adv_x/dx).eval() + Eigen::kroneckerProduct(velocity*Adv_y/dy, I_N).eval();
    }

    //! Destructor
    ~RHS_Dif_Adv_2D() {}
};

//? ====================================================================================== ?//