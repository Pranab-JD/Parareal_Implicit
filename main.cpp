#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

//? LeXInt Timer and functions
#include "Timer.hpp"
#include "functions.hpp"

//? Problems
#include "Dif_Adv_2D.hpp"

//? Solvers
#include "Implicit.hpp"

//? Eigen
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;

//? ====================================================================================== ?//

int main(int argc, char** argv)
{

    int index = atoi(argv[1]);              // N = 2^index * 2^index
    double n_cfl = atof(argv[2]);           // dt = n_cfl * dt_cfl
    double tol = atof(argv[3]);             // User-specified tolerance
    int num_time_steps = atoi(argv[4]);     // Final simulation time
    
    string integrator = "Implicit_Euler";   // Integrator
    if (argc >= 6)
        integrator = argv[5];

    string movie = "no";                    // Default param = "no"
    if (argc == 7)
        movie = argv[6];                    // Set to "yes" to write data for plots/movie

    int num_threads;                        // # of OpenMP threads
    #pragma omp parallel
    num_threads = omp_get_num_threads();

    //! Set GPU spport to false
    bool GPU_access = false;

    //* Initialise parameters
    int n = pow(2, index);                          // # grid points (1D)
    int N = n*n;                                    // # grid points (2D)
    double xmin = -1;                               // Left boundary (limit)
    double xmax =  1;                               // Right boundary (limit)
    double ymin = -1;                               // Left boundary (limit)
    double ymax =  1;                               // Right boundary (limit)
    vector<double> X(n);                            // Array of grid points
    vector<double> Y(n);                            // Array of grid points
    Eigen::VectorXd u(N);                           // Initial condition
    Eigen::VectorXd u_sol(N);                       // Solution vector 

    //* Set up X, Y arrays and initial condition
    for (int ii = 0; ii < n; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/n;
        Y[ii] = ymin + ii*(ymax - ymin)/n;
    }

    //* Initialise additional parameters
    double dx = X[2] - X[1];                              // Grid spacing
    double dy = Y[2] - Y[1];                              // Grid spacing
    double velocity = 10;                                   // Advection speed

    //* Temporal parameters
    double dif_cfl = (dx*dx * dy*dy)/(2*dx*dx + 2*dy*dy);   // Diffusion CFL
    double adv_cfl = min(dx/velocity, dy/velocity);         // Advection CFL
    double dt = n_cfl*min(dif_cfl, adv_cfl);                // Step size

    double time = 0;                                        // Simulation time elapsed                          
    int time_steps = 0;                                     // # time steps
    int iters = 0;                                          // # of iterations per time step
    int iters_total = 0;                                    // Total # of iterations during the simulation

    cout << endl << "N = " << N << ", tol = " << tol << ", Time steps = " << num_time_steps << endl;
    cout << "N_cfl = " << n_cfl << ", CFL: " << min(dif_cfl, adv_cfl) << ", dt = " << dt << endl << endl;

    //? Identity matrix
    Eigen::SparseMatrix<double> I_N(n, n);
    I_N.setIdentity();

    //? Diffusion
    Eigen::SparseMatrix<double> Dif_xx(n, n);
    Eigen::SparseMatrix<double> Dif_yy(n, n);
    Eigen::SparseMatrix<double> A_dif(N, N);

    //? Advection
    Eigen::SparseMatrix<double> Adv_x(n, n);
    Eigen::SparseMatrix<double> Adv_y(n, n);
    Eigen::SparseMatrix<double> A_adv(N, N);

    //? Create an instance of the RHS class
    RHS_Dif_Adv_2D RHS(n, dx, dy, velocity);
    RHS.Diffusion_matrix(Dif_xx, Dif_yy, I_N, A_dif);
    RHS.Advection_matrix(Adv_x, Adv_y, I_N, A_adv);

    //? Choose problem
    string problem = "Diff_Adv_2D";
    Eigen::SparseMatrix<double> A_diff_adv(N, N);       //* Add diffusion and advection matrices 
    A_diff_adv = A_diff_adv + A_adv;
   
    //! Print the matrix (avoid doing this for large matrices)
    // cout << "Diffusion matrix:" << endl << Eigen::MatrixXd(A_dif) << endl << endl;
    // cout << "Advection matrix:" << endl << Eigen::MatrixXd(A_adv) << endl << endl;

    if (problem == "Diff_Adv_2D")
    {
        //? Initial condition
        for (int ii = 0; ii < n; ii++)
        {
            for (int jj = 0; jj< n; jj++)
            {
                u(n*ii + jj) = 1 + exp(-((X[ii] + 0.5)*(X[ii] + 0.5) + (Y[jj] + 0.5)*(Y[jj] + 0.5))/0.01);
            }
        }
    }
    else
    {
        cout << "Undefined problem! Please check that you have entered the correct problem." << endl;
    } 

    //! Create directories (for movies)
    if (movie == "yes")
    {
        int sys_value = system(("mkdir -p ./movie/"));
        string directory = "./movie/";
    }

    //? Create matrices and vectors for implicit integrators
    Eigen::VectorXd rhs_vector(N);
    Eigen::SparseMatrix<double> LHS_matrix(n, n);

    //! Time Loop
    LeXInt::timer time_loop;
    time_loop.start();

    cout << "Running the 2D diffusion--advection problem with the " << integrator << " integrator." << endl << endl;

    for (int nn = 0; nn < num_time_steps; nn++)
    {
        //? ------------- List of integrators ------------- ?//

        if (integrator == "Implicit_Euler")
        {
            implicit_Euler(A_diff_adv, u, tol, dt, iters, LHS_matrix, rhs_vector);
        }
        else if (integrator == "CN")
        {
            //TODO: Implement CN
        }
        else
        {
            cout << "Incorrect integrator! Please recheck. Terminating simulations ... " << endl << endl;
            return 1;
        }

        //? ----------------------------------------------- ?//

        //* Update variables
        time = time + dt;
        time_steps = time_steps + 1;
        iters_total = iters_total + iters;
        
        //! The solution (u) does not have to updated explcitly.
        //! The integrators automatically updates the solution.

        if (time_steps % 5 == 0)
        {
            cout << "Time step      : " << time_steps << endl;
            cout << "Simulation time: " << time << endl << endl;
        }

        //! Write data to files (for movies)
        if (time_steps % 5 == 0 && movie == "yes")
        {
            string output_data = "./movie/" +  to_string(time_steps) + ".txt";
            ofstream data;
            data.open(output_data); 
            for(int ii = 0; ii < N; ii++)
            {
                data << setprecision(16) << u[ii] << endl;
            }
            data.close();
        }
    }

    time_loop.stop();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time            : " << time << endl;
    cout << "Total number of time steps : " << time_steps << endl;
    cout << "Total number of iterations : " << iters_total << endl;
    cout << "Total time elapsed (s)     : " << time_loop.total() << endl;
    cout << "==================================================" << endl << endl;


    //? Create directory to write simulation results/parameters
    int sys_value = system(("mkdir -p ./" + integrator + "/cores_" + to_string(num_threads)).c_str());
    string directory = "./" + integrator + "/cores_" + to_string(num_threads);
    string results = directory + "/Parameters.txt";
    ofstream params;
    params.open(results);
    params << "Grid points: " << N << endl;
    params << "Step size: " << dt << endl;
    params << "Tolerance (for implicit methods): " << tol << endl;
    params << "Simulation time: " << time << endl;
    params << "Total number of time steps: " << time_steps << endl;
    params << "Number of OpenMP threads: " << num_threads << endl;
    params << endl;
    params << "Total iterations (for implicit methods): " << iters_total << endl;
    params << setprecision(16) << "Runtime (s): " << time_loop.total() << endl;
    params.close();

    cout << "Simulations complete!" << endl;

    return 0;
}