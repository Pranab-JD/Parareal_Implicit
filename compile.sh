#!/bin/bash

rm -rf Parareal_Implicit

### Compile using g++
g++ ../main.cpp -O3 -fopenmp -I ../../eigen/ -o Parareal_Implicit_2D

g++ ../main_1D.cpp -O3 -fopenmp -I ../../eigen/ -o Parareal_Implicit_1D