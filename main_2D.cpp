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
#include "Parareal.hpp"

//? Eigen
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <unsupported/Eigen/KroneckerProduct>

using namespace std;

//? ====================================================================================== ?//

int main(int argc, char** argv)
{
    //? Input parameters

    int index = atoi(argv[1]);              // N = 2^index * 2^index
    double n_cfl = atof(argv[2]);           // dt = n_cfl * dt_cfl
    double tol = atof(argv[3]);             // User-specified tolerance
    int num_time_steps = atoi(argv[4]);     // Final simulation time
    string integrator = argv[5];            // Integrator
    string movie = argv[6];                 // Write data to files every N time steps: "yes" or "no"

    int num_threads;                        // # of OpenMP threads
    #pragma omp parallel
    num_threads = omp_get_num_threads();

    //! Set GPU spport to false
    bool GPU_access = false;

    //* ----------------------------------------------------------------- *//

    //* Initialise parameters
    int n = pow(2, index);                          // # grid points (1D)
    int N = n*n;                                    // # grid points (2D)
    double xmin = -1;                               // Left boundary (limit)
    double xmax =  1;                               // Right boundary (limit)
    double ymin = -1;                               // Left boundary (limit)
    double ymax =  1;                               // Right boundary (limit)
    vector<double> X(n);                            // Array of grid points
    vector<double> Y(n);                            // Array of grid points
    Eigen::VectorXd u_init(N);                      // Initial condition

    //* Set up X, Y arrays and initial condition
    #pragma omp parallel for
    for (int ii = 0; ii < n; ii++)
    {
        X[ii] = xmin + ii*(xmax - xmin)/n;
        Y[ii] = ymin + ii*(ymax - ymin)/n;
    }

    //* Initialise additional parameters
    double dx = X[2] - X[1];                              // Grid spacing
    double dy = Y[2] - Y[1];                              // Grid spacing
    double velocity = 0.2;                                 // Advection speed

    //* Temporal parameters
    double dif_cfl = (dx*dx * dy*dy)/(2*dx*dx + 2*dy*dy);   // Diffusion CFL
    double adv_cfl = min(dx/velocity, dy/velocity);         // Advection CFL
    double dt = n_cfl*min(dif_cfl, adv_cfl);                // Step size
    double T_final = num_time_steps * dt;

    double time = 0;                                        // Simulation time elapsed                          
    int time_steps = 0;                                     // # time steps
    int iters = 0;                                          // # of iterations per time step
    int iters_serial = 0;                                   // Total # of iterations during the simulation

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
    A_diff_adv = A_dif + A_adv;
   
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
                u_init(n*ii + jj) = 1 + 10*exp(-((X[ii] + 0.5)*(X[ii] + 0.5) + (Y[jj] + 0.5)*(Y[jj] + 0.5))/0.02);
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

    LeXInt::timer time_loop, serial_time;
    LeXInt::timer parareal_time;

    //? ================================================================================= ?//

    //! Time Loop (Serial Implementation)
    time_loop.start();

    cout << "Running the 2D diffusion--advection problem with the " << integrator << " integrator." << endl << endl;

    serial_time.start();

    Eigen::VectorXd u(u_init);

    for (int nn = 0; nn < num_time_steps; nn++)
    {
        //? ------------- List of integrators ------------- ?//

        if (integrator == "Implicit_Euler")
        {
            implicit_Euler(A_diff_adv, u, tol, dt, iters, LHS_matrix, rhs_vector);
        }
        else if (integrator == "CN")
        {
            CN(A_diff_adv, u, tol, dt, iters, LHS_matrix, rhs_vector);
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
        iters_serial = iters_serial + iters;
        
        //! The solution (u) does not have to updated explicitly.
        //! The integrators automatically updates the solution.

        if (time_steps % 100 == 0)
        {
            cout << "Time step      : " << time_steps << endl;
            cout << "Simulation time: " << time << endl << endl;
        }

        //! Write data to files (for movies)
        if (time_steps % 100 == 0 && movie == "yes")
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

    serial_time.stop();

    // cout << "# of GMRes iterations for serial   : " << iters_serial << endl;
    cout << "Time elapsed for serial (s)        : " << serial_time.total() << endl << endl;

    //? ================================================================================= ?//

    //! Parareal
    cout << "Running the 2D diffusion--advection problem with the Parareal algorithm." << endl << endl;

    //? num_threads = num_time_steps = max_parareal_iters
    int num_coarse_steps = num_threads;
    int num_fine_steps_per_coarse = atoi(argv[7]);

    int iters_parareal;
    Eigen::VectorXd u_parareal = u_init;
    
    cout << "Number of coarse time steps:   " << num_coarse_steps << endl;
    cout << "Number of fine steps per coarse step:   " << num_fine_steps_per_coarse << endl;

    parareal_time.start();
    parareal(A_diff_adv, u_parareal, tol, iters_parareal, LHS_matrix, rhs_vector, T_final, num_coarse_steps, num_fine_steps_per_coarse);
    parareal_time.stop();

    // cout << "# of GMRes iterations for parareal   : " << iters_parareal << endl;
    cout << "Total time elapsed for parareal (s)  : " << parareal_time.total() << endl << endl;
    cout << "Speedup (serial to parareal) : " << serial_time.total()/parareal_time.total() << endl;       

    //? ================================================================================= ?//

    Eigen::VectorXd u_diff = u - u_parareal;
    cout << endl << "Error wrt to serial: " << u_diff.norm()/u.norm() << endl;

    time_loop.stop();

    cout << endl << "==================================================" << endl;
    cout << "Simulation time                         : " << time << endl;
    cout << "Total number of time steps (serial)     : " << time_steps << endl;
    cout << "Total time elapsed (s)                  : " << time_loop.total() << endl;
    cout << "==================================================" << endl << endl;

    //? ================================================================================= ?//

    //? Create directory to write simulation results/parameters
    int sys_value = system(("mkdir -p ./" + integrator + "/cores_" + to_string(num_threads)).c_str());
    string directory = "./" + integrator + "/cores_" + to_string(num_threads);
    string results = directory + "/Params_dt_cfl_" + to_string(n_cfl) + ".txt";
    ofstream params;
    params.open(results);
    params << "Grid points: " << N << endl;
    params << "Step size: " << dt << endl;
    params << "Tolerance (for iterative methods): " << tol << endl;
    params << "Simulation time: " << time << endl;
    params << endl;
    params << "Serial" << endl;
    params << "Total number of time steps: " << time_steps << endl;
    params << "Total iterations (serial): " << iters_serial << endl;
    params << "Serial Runtime (s): " << serial_time.total() << endl;
    params << endl;
    params << "Parareal " << endl;
    params << "Number of OpenMP threads: " << num_threads << endl;
    params << "Number of coarse time steps: " << num_coarse_steps << endl;
    params << "Number of fine steps per coarse step: " << num_fine_steps_per_coarse << endl;
    params << "Total iterations (parareal): " << iters_parareal << endl;
    params << "Parareal Runtime (s): " << parareal_time.total() << endl;
    params << "Speedup (serial to parareal) : " << serial_time.total()/parareal_time.total() << endl;
    params.close();

    //? Create directory to write final simulation data
    string final_data = directory + "/dt_cfl_" + to_string(n_cfl) + "_serial.txt";
    ofstream data;
    data.open(final_data); 
    for(int ii = 0; ii < N; ii++)
    {
        data << setprecision(16) << u[ii] << endl;
    }
    data.close();

    final_data = directory + "/dt_cfl_" + to_string(n_cfl) + "_parareal.txt";
    data.open(final_data); 
    for(int ii = 0; ii < N; ii++)
    {
        data << setprecision(16) << u_parareal[ii] << endl;
    }
    data.close();

    cout << "Simulations complete!" << endl;

    return 0;
}