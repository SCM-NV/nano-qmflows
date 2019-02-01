#!/bin/bash

g++ -O2 -I$HOME/modules/libint/include -I/usr/include/eigen3 compute_integrals.cc $HOME/modules/libint/lib/libint2.a  -o main
