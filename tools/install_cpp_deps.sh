# Download mininconda and setup a valid(ish) nano-qmflows environment
# with all C++ dependencies.

PYTHON_VERSION="$1"

wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod u+rx miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda install -c conda-forge python=$PYTHON_VERSION boost eigen libint==2.6.0 highfive -y
