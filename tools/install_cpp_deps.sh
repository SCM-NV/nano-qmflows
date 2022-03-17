#!/bin/bash

echo
set -e

N_PROC=2

BOOST_VERSION="1.78.0"
EIGEN_VERSION="3.4.0"
LIBINT_VERSION="2.6.0"
HDF5_VERSION="1.12.1"
GMP_VERIOSN="6.2.1"
HIGHFIVE_VERSION="2.3.1"

BOOST_VERSION_UNDERSCORE="${BOOST_VERSION//./_}"
HDF5_VERSION_SHORT="${HDF5_VERSION%.*}"

echo "Setting up C++ dependencies in $PWD"

setup_boost () {
    start=$SECONDS
    echo ::group::"Download boost $BOOST_VERSION"
    if [ ! -d "boost" ]; then
        curl -Ls https://boostorg.jfrog.io/artifactory/main/release/$BOOST_VERSION/source/boost_$BOOST_VERSION_UNDERSCORE.tar.gz -o boost_$BOOST_VERSION_UNDERSCORE.tar.gz
        tar -xf boost_$BOOST_VERSION_UNDERSCORE.tar.gz
        mv boost_$BOOST_VERSION_UNDERSCORE boost
        rm boost_$BOOST_VERSION_UNDERSCORE.tar.gz
    fi
    export BOOST_DIR="$PWD/boost"
    echo ::endgroup::
    printf "%71.71s\n" "✓ $(($SECONDS - $start))s"
}

setup_eigen () {
    start=$SECONDS
    echo ::group::"Download eigen $EIGEN_VERSION"
    if [ ! -d "eigen" ]; then
        curl -s https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.gz -o eigen-$EIGEN_VERSION.tar.gz
        tar -xf eigen-$EIGEN_VERSION.tar.gz
        mv eigen-$EIGEN_VERSION eigen
        rm eigen-$EIGEN_VERSION.tar.gz
    fi
    echo ::endgroup::
    printf "%71.71s\n" "✓ $(($SECONDS - $start))s"
}

setup_gmp () {
    start=$SECONDS
    echo ::group::"Download GMP $GMP_VERSION"
    if [ ! -d "gmp" ]; then
        curl -Ls https://gmplib.org/download/gmp/gmp-$GMP_VERSION.tar.xz -o gmp-$GMP_VERSION.tar.xz
        tar -xf gmp-$GMP_VERSION.tar.xz
        mkdir gmp
        GMP_DIR="$PWD/gmp"
        echo ::endgroup::
        printf "%71.71s\n" "✓ $(($SECONDS - $start))s"

        start=$SECONDS
        echo ::group::"Configure GMP $GMP_VERSION"
        cd gmp-$GMP_VERSION
        ./configure --prefix=$GMP_DIR --enable-cxx
        echo ::endgroup::
        printf "%71.71s\n" "✓ $(($SECONDS - $start))s"

        start=$SECONDS
        echo ::group::"Build GMP $GMP_VERSION"
        make -j $N_PROC
        make install
        cd ..
        rm gmp-$GMP_VERSION.tar.xz
        rm -rf gmp-$GMP_VERSION
    fi
    echo ::endgroup::
    printf "%71.71s\n" "✓ $(($SECONDS - $start))s"
}

setup_libint () {
    start=$SECONDS
    echo ::group::"Download libint $LIBINT_VERSION"
    if [ ! -d "libint2" ]; then
        curl -Ls https://github.com/evaleev/libint/archive/refs/tags/v$LIBINT_VERSION.tar.gz -o libint-$LIBINT_VERSION.tar.gz
        tar -xf libint-$LIBINT_VERSION.tar.gz
        mkdir libint2
        mkdir libint_build
        LIBINT_DIR="$PWD/libint2"
        echo ::endgroup::
        printf "%71.71s\n" "✓ $(($SECONDS - $start))s"

        start=$SECONDS
        echo ::group::"Configure libint $LIBINT_VERSION"
        cd libint-$LIBINT_VERSION
        chmod u+rx autogen.sh
        ./autogen.sh
        cd ../libint_build
        ../libint-$LIBINT_VERSION/configure --enable-shared=yes --prefix=$LIBINT_DIR CPPFLAGS="-I$BOOST_DIR -I$GMP_DIR/include"
        echo ::endgroup::
        printf "%71.71s\n" "✓ $(($SECONDS - $start))s"

        start=$SECONDS
        echo ::group::"Build libint $LIBINT_VERSION"
        make -j $N_PROC
        make install
        cd ..
        rm libint-$LIBINT_VERSION.tar.gz
        rm -rf libint-$LIBINT_VERSION
        rm -rf libint_build
    fi
    echo ::endgroup::
    printf "%71.71s\n" "✓ $(($SECONDS - $start))s"
}

setup_hdf5 () {
    start=$SECONDS
    echo ::group::"Download HDF5 $HDF5_VERSION"
    if [ ! -d "hdf5" ]; then
        curl -s https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$HDF5_VERSION_SHORT/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz -o hdf5-$HDF5_VERSION.tar.gz
        tar -xf hdf5-$HDF5_VERSION.tar.gz
        HDF5_DIR="$PWD/hdf5"
        mkdir hdf5
        echo ::endgroup::
        printf "%71.71s\n" "✓ $(($SECONDS - $start))s"

        start=$SECONDS
        echo ::group::"Configure HDF5 $HDF5_VERSION"
        cd hdf5-$HDF5_VERSION
        chmod u+rx autogen.sh
        ./configure --prefix="$HDF5_DIR" --enable-build-mode=production
        echo ::endgroup::
        printf "%71.71s\n" "✓ $(($SECONDS - $start))s"

        start=$SECONDS
        echo ::group::"Build HDF5 $HDF5_VERSION"
        make -j $N_PROC
        make install
        cd ..
        rm hdf5-$HDF5_VERSION.tar.gz
        rm -rf hdf5-$HDF5_VERSION
    fi
    echo ::endgroup::
    printf "%71.71s\n" "✓ $(($SECONDS - $start))s"
}

setup_highfive () {
    start=$SECONDS
    echo ::group::"Download highfive $HIGHFIVE_VERSION"
    if [ ! -d "highfive" ]; then
        curl -Ls https://github.com/BlueBrain/HighFive/archive/refs/tags/v$HIGHFIVE_VERSION.tar.gz -o highfive-$HIGHFIVE_VERSION.tar.gz
        tar -xf highfive-$HIGHFIVE_VERSION.tar.gz
        mv HighFive-$HIGHFIVE_VERSION highfive
        rm highfive-$HIGHFIVE_VERSION.tar.gz
    fi
    echo ::endgroup::
    printf "%71.71s\n" "✓ $(($SECONDS - $start))s"
}

setup_boost
setup_eigen
setup_gmp
setup_libint
setup_hdf5
setup_highfive
