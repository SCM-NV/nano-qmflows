#!/bin/bash

set -e

setup_boost () {
    echo "Downloading boost"
    curl -L https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz -o boost_1_78_0.tar.gz
    tar -xf boost_1_78_0.tar.gz
    mv boost_1_78_0 boost
    export BOOSt_INCLUDE_DIR="$PWD/boost/boost"
}

setup_libint () {
    echo "Downloading libint"
    curl -L https://github.com/evaleev/libint/archive/refs/tags/v2.6.0.tar.gz -o libint-2.6.0.tar.gz
    tar -xf libint-2.6.0.tar.gz
    mkdir libint
    export LIBINT_DIR="$PWD/libint"
    export LIBINT_INCLUDE_DIR="$LIBINT_DIR/include"
    export LIBINT_LIB_DIR="$LIBINT_DIR/lib"

    echo "Building libint"
    cd libint-2.6.0
    chmod u+rx autogen.sh
    ./autogen.sh
    ./configure --prefix=$LIBINT_DIR CPPFLAGS="-I/$BOOSt_INCLUDE_DIR"
    make -j 2
    make install
    cd ..
}

setup_eigen () {
    echo "Downloading eigen"
    curl https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -o eigen-3.4.0.tar.gz
    tar -xf eigen-3.4.0.tar.gz
    mv eigen-3.4.0 eigen
    export EIGEN3_INCLUDE_DIR="$PWD/eigen/Eigen"
}

setup_hdf5 () {
    echo "Downloading HDF5"
    curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz -o hdf5-1.12.1.tar.gz
    tar -xf hdf5-1.12.1.tar.gz
    mkdir hdf5
    export HDF5_DIR="$PWD/hdf5"

    echo "Building HDF5"
    cd hdf5-1.12.1
    chmod u+rx autogen.sh
    ./configure --prefix=$HDF5_DIR --enable-build-mode=production
    make -j 2
    make install
    cd ..
}

setup_highfive () {
    echo "Downloading highfive"
    curl -L https://github.com/BlueBrain/HighFive/archive/refs/tags/v2.3.1.tar.gz -o highfive-2.3.1.tar.gz
    tar -xf highfive-2.3.1.tar.gz
    mv highfive-2.3.1 highfive
}

cd ..
setup_boost
setup_libint
setup_eigen
setup_hdf5
setup_highfive
