import pytest
from packaging.version import Version
from assertionlib import assertion

import nanoqm


def test_version() -> None:
    """Check that the nano-qmflows version is PEP 440 compliant."""
    assertion.assert_(Version, nanoqm.__version__)


@pytest.mark.parametrize("name", ["major", "minor", "micro"])
def test_version_info(name: str) -> None:
    attr = getattr(nanoqm.version_info, name)
    assertion.isinstance(attr, int)
    assertion.ge(attr, 0)
