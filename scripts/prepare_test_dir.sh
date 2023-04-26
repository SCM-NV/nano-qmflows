# Bash script for creating a more isolated testing environment

# In practice this just copies a bunch of test files in order to
# avoid intermingling between the nano-qmflows root files and the location
# where it is actually installed, as this is a recipe for disaster when
# extension modules are involved

set -euo pipefail

mkdir /tmp/nanoqm
cp conftest.py /tmp/nanoqm/
cp -r test /tmp/nanoqm/
cp CITATION.cff /tmp/nanoqm/
cp codecov.yaml /tmp/nanoqm/
