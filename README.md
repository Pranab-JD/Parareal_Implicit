# Parareal Implicit
This code uses the [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for matrices, vectors, and GMRes.

## Installation
1. Install Eigen
``` shell
git clone https://gitlab.com/libeigen/eigen.git
```
2. Install Parareal Implicit (Note: Eigen and Parareal_Implicit should be in the same directory)
``` shell
git clone https://github.com/Pranab-JD/Parareal_Implicit.git
```
3. Create build directory
``` shell
cd Parareal_Implicit && mkdir build && cd build
```
4. Compile the code
``` shell
bash ../compile.sh
```
5. Run the code
``` shell
bash ./run.sh
```

### Info on input arguments:
Argument 1: The executable (program) to be run (./Parareal_Implicit) <br />
Argument 2: Grid points along X & Y (8 --> 2^8 x 2^8) <br />
Argument 3: Time step size in terms of n_cfl x dt_cfl (2.3 --> 2.3 * dt_cfl; dt_cfl is computed in main.cpp) <br />
Argument 4: Tolerance, for iterative methods (e.g., 1e-10) <br />
Argument 5: Simulation time (1.2 --> T_final = 1.2; NOT wall-clock time) <br />
Argument 6: Write data to files every N time steps for movies (if "yes", will create files & write data) <br />
