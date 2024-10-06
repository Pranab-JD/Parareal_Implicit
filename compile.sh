#!/bin/bash

### Compile using g++
g++ ../main.cpp -O3 -fopenmp -I ../../eigen/ -o Parareal_Implicit
