import re
from unittest import mock

import setuptools
import pkg_resources
from assertionlib import assertion

PATTERN = re.compile("@ ")


def test_requirements_txt() -> None:
    """Test that ``test_requirements.txt`` and ``setup.py`` are synced."""
    with open("test_requirements.txt", "r") as f:
        requirements_set = {PATTERN.sub("@", str(i)) for i in pkg_resources.parse_requirements(f)}

    with mock.patch.object(setuptools, "setup") as mock_setup:
        import setup
    _, kwargs = mock_setup.call_args

    setup_set = set(kwargs["install_requires"])
    setup_set.update(kwargs["extras_require"]["test"])

    diff_set = requirements_set ^ setup_set
    assertion.not_(diff_set, message="Dependency mismatch between setup.py and test_requirements.txt")
