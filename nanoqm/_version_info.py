"""The Nano-QMFlows version tuple."""

from nanoutils import VersionInfo
from packaging.version import Version

from ._version import __version__

__all__ = ["version_info"]

VERSION = Version(__version__)
version_info = VersionInfo._make(VERSION.release[:3])
