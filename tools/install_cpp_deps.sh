#!/bin/bash

set -e

BOOST_VERSION="1.78.0"
EIGEN_VERSION="3.4.0"
LIBINT_VERSION="2.6.0"
HDF5_VERSION="1.12.1"
HIGHFIVE_VERSION="2.3.1"

BOOST_VERSION_UNDERSCORE="${BOOST_VERSION//./_}"
HDF5_VERSION_SHORT="${HDF5_VERSION%.*}"

setup_boost () {
    start=$SECONDS

    echo ::group::"Download boost"
    curl -L https://boostorg.jfrog.io/artifactory/main/release/$BOOST_VERSION/source/boost_$BOOST_VERSION_UNDERSCORE.tar.gz -o boost_$BOOST_VERSION.tar.gz
    tar -xf boost_$BOOST_VERSION.tar.gz
    mv boost_$BOOST_VERSION_UNDERSCORE boost
    export BOOST_INCLUDEDIR="$PWD/boost/boost"
    echo ::endgroup::

    printf "%66.66s\n" "✓ $(($SECONDS - $start))s"
}

setup_eigen () {
    start=$SECONDS

    echo ::group::"Download eigen"
    curl https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz -o eigen-$EIGEN_VERSION.tar.gz
    tar -xf eigen-$EIGEN_VERSION.tar.gz
    mv eigen-$EIGEN_VERSION eigen
    export EIGEN3_INCLUDEDIR="$PWD/eigen/Eigen"
    echo ::endgroup::

    printf "%66.66s\n" "✓ $(($SECONDS - $start))s"
}

setup_libint () {
    start=$SECONDS

    echo ::group::"Download libint"
    curl -L https://github.com/evaleev/libint/archive/refs/tags/v$LIBINT_VERSION.tar.gz -o libint-$LIBINT_VERSION.tar.gz
    tar -xf libint-$LIBINT_VERSION.tar.gz
    mv libint-$LIBINT_VERSION libint
    export LIBINT_INCLUDEDIR="$PWD/libint/include"
    echo ::endgroup::

    printf "%66.66s\n" "✓ $(($SECONDS - $start))s"
}

setup_hdf5 () {
    start=$SECONDS
    echo ::group::"Download HDF5"
    curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$HDF5_VERSION_SHORT/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz -o hdf5-$HDF5_VERSION.tar.gz
    tar -xf hdf5-$HDF5_VERSION.tar.gz
    mkdir hdf5
    export HDF5_DIR="$PWD/hdf5"
    export HDF5_INCLUDEDIR="$HDF5_DIR/include"
    export HDF5_LIBDIR="$HDF5_DIR/lib"
    echo ::endgroup::

    printf "%66.66s\n" "✓ $(($SECONDS - $start))s"
    start=$SECONDS

    echo ::group::"Build HDF5"
    cd hdf5-$HDF5_VERSION
    chmod u+rx autogen.sh
    ./configure --prefix=$HDF5_DIR --enable-build-mode=production
    make -j 2
    make install
    cd ..
    echo ::endgroup::

    printf "%66.66s\n" "✓ $(($SECONDS - $start))s"
}

setup_highfive () {
    start=$SECONDS

    echo ::group::"Download highfive"
    curl -L https://github.com/BlueBrain/HighFive/archive/refs/tags/v$HIGHFIVE_VERSION.tar.gz -o highfive-$HIGHFIVE_VERSION.tar.gz
    tar -xf highfive-$HIGHFIVE_VERSION.tar.gz
    mv highfive-$HIGHFIVE_VERSION highfive
    export HIGHFIVE_INCLUDEDIR="$PWD/highfive/include"
    echo ::endgroup::

    printf "%66.66s\n" "✓ $(($SECONDS - $start))s"
}

cd ..
setup_boost
setup_eigen
setup_libint
setup_hdf5
setup_highfive

export QMFLOWS_INCLUDEDIR="$HIGHFIVE_INCLUDEDIR:$HDF5_INCLUDEDIR:$LIBINT_INCLUDEDIR:$EIGEN3_INCLUDEDIR:$BOOST_INCLUDEDIR"
export QMFLOWS_LIBDIR="$LIBINT_INCLUDEDIR"
