# Bash script for downloading CP2K

set -euo pipefail

ARCH="$1"
VERSION="$2"

# After the 9.1 release CP2K switched to a `<year>.<suffix>` version scheme (e.g. `2021.1`)
if [[ $VERSION =~ [0-9][0-9][0-9][0-9]\.[0-9]+ ]]; then
    PLAT="Linux-gnu"
    VERSION_LONG=v"$VERSION"
else
    PLAT="Linux"
    VERSION_LONG=v"$VERSION".0
fi

echo "Installing CP2K $ARCH $VERSION binaries"
curl -Lsf https://github.com/cp2k/cp2k/releases/download/$VERSION_LONG/cp2k-$VERSION-$PLAT-$ARCH.ssmp -o /usr/local/bin/cp2k.ssmp
chmod u+rx /usr/local/bin/cp2k.ssmp
