#!/bin/bash

set -e

setup_boost () {
    echo "Downloading boost"
    curl -Ls https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz -o boost_1_78_0.tar.gz
    tar -xf boost_1_78_0.tar.gz
    mv boost_1_78_0 boost
    export BOOST_INCLUDEDIR="$PWD/boost/boost"
    echo "Boost setup successful"
}

setup_eigen () {
    echo "Downloading eigen"
    curl -s https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -o eigen-3.4.0.tar.gz
    tar -xf eigen-3.4.0.tar.gz
    mv eigen-3.4.0 eigen
    export EIGEN3_INCLUDEDIR="$PWD/eigen/Eigen"
    echo "Eigen setup successful"
}

setup_libint () {
    echo "Downloading libint"
    curl -Ls https://github.com/evaleev/libint/archive/refs/tags/v2.6.0.tar.gz -o libint-2.6.0.tar.gz
    tar -xf libint-2.6.0.tar.gz
    mv libint-2.6.0 libint
    export LIBINT_INCLUDEDIR="$PWD/libint/include"
    echo "Libint setup successful"
}


setup_highfive () {
    echo "Downloading highfive"
    curl -Ls https://github.com/BlueBrain/HighFive/archive/refs/tags/v2.3.1.tar.gz -o highfive-2.3.1.tar.gz
    tar -xf highfive-2.3.1.tar.gz
    mv highfive-2.3.1 highfive
    export HIGHFIVE_INCLUDEDIR="$PWD/highfive/include"
    echo "Highfive setup successful"
}

setup_hdf5 () {
    echo "Downloading HDF5"
    curl -s https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz -o hdf5-1.12.1.tar.gz
    tar -xf hdf5-1.12.1.tar.gz
    mkdir hdf5
    export HDF5_DIR="$PWD/hdf5"
    export HDF5_INCLUDEDIR="$HDF5_DIR/include"
    export HDF5_LIBDIR="$HDF5_DIR/lib"

    echo "Building HDF5"
    cd hdf5-1.12.1
    chmod u+rx autogen.sh
    ./configure --prefix=$HDF5_DIR --enable-build-mode=production
    make -j 2
    make install
    cd ..
    echo "HDF5 setup successful"
}

cd ..
setup_boost
setup_eigen
setup_libint
setup_hdf5
setup_highfive

export QMFLOWS_INCLUDEDIR="$HIGHFIVE_INCLUDEDIR:$HDF5_INCLUDEDIR:$LIBINT_INCLUDEDIR:$EIGEN3_INCLUDEDIR:$BOOST_INCLUDEDIR"
export QMFLOWS_LIBDIR="$LIBINT_INCLUDEDIR"
