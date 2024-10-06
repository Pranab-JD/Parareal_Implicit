# Pararel Implicit
This code uses the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for matrices, vectors, and GMRes.

### Info on input arguments:
Argument 1: The executable (program) to be run (./Parareal_Implicit)

Argument 2: Grid points along X & Y (8 --> 2^8 x 2^8)

Argument 3: Time step size in terms of n_cfl x dt_cfl (2.3 --> 2.3 * dt_cfl; dt_cfl is computed in main.cpp)

Argument 4: Tolerance, for iterative methods (1e-10)

Argument 5: Simulation time (1.2 --> T_final = 1.2; NOT wall-clock time)

Argument 6: Write data to files every N time steps for movies (if "yes", will create files & write data)
