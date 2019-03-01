#!/bin/bash

export MINICONDA=${HOME}/miniconda3/envs/namd

# CPP call
g++ -O2 -o main -fopenmp -I${MINICONDA}/include -I${MINICONDA}/include/eigen3  -I${MINICONDA}/include/python3.6m -Iinclude compute_integrals.cc -L${MINICONDA}/lib -lhdf5 -lint2
