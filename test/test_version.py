from packaging.version import Version
import nanoqm


def test_version() -> None:
    """Check that the nano-qmflows version is PEP 440 compliant."""
    assert Version(nanoqm.__version__)
