import pytest
from packaging.version import Version
from assertionlib import assertion

import nanoqm


def test_version() -> None:
    """Check that the nano-qmflows version is PEP 440 compliant."""
    assertion.assert_(Version, nanoqm.__version__)


def test_dev_version(is_release: bool) -> None:
    if not is_release:
        pytest.skip("Requires a nano-qmflows release")
        return None
    version = Version(nanoqm.__version__)
    assertion.not_(version.is_devrelease)
