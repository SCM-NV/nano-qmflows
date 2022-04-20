# Bash script for downloading CP2K

set -euo pipefail

VERSION="$1"
VERSION_LONG=v"$VERSION".0

echo "Installing CP2K $VERSION binaries"
curl -Ls https://github.com/cp2k/cp2k/releases/download/$VERSION_LONG/cp2k-$VERSION-Linux-x86_64.ssmp -o cp2k.ssmp
chmod u+rx cp2k.ssmp
mv cp2k.ssmp /usr/local/bin/cp2k.popt
