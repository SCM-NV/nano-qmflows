# Download mininconda and setup a valid(ish) nano-qmflows environment
# with all C++ dependencies.

PYTHON_VERSION="$1"

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
chmod u+rx miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda install -c conda-forge python=$PYTHON_VERSION boost==1.70 eigen libint==2.6.0 highfive -y
