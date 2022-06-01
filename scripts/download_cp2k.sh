# Bash script for downloading CP2K

set -euo pipefail

ARCH="$1"
VERSION="$2"
VERSION_LONG=v"$VERSION".0

echo "Installing CP2K $ARCH $VERSION binaries"
curl -Lsf https://github.com/cp2k/cp2k/releases/download/$VERSION_LONG/cp2k-$VERSION-Linux-$ARCH.ssmp -o /usr/local/bin/cp2k.ssmp
chmod u+rx /usr/local/bin/cp2k.ssmp
