#!/bin/bash

# CPP call
g++ -O2 -I$HOME/modules/libint/include -I/usr/include/eigen3 compute_integrals.cc $HOME/modules/libint/lib/libint2.a  -o main

# Python bind
g++ -O2  -shared -std=c++14 -I$HOME/modules/libint/include -I/usr/include/eigen3 $HOME/modules/libint/lib/libint2.a -fPIC `python3 -m pybind11 --includes` compute_integrals.cc -o compute_integrals`python3-config --extension-suffix`
