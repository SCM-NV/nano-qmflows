import re
from unittest import mock
from typing import Dict, Any

import pytest
import setuptools
import pkg_resources
from assertionlib import assertion

PATTERN = re.compile("@ ")


@pytest.fixture(scope="module", autouse=True, name="setup_kwargs")
def get_setup_kwargs() -> Dict[str, Any]:
    """Get the nano-qmflows ``setup()`` content as a dictionary."""
    with mock.patch.object(setuptools, "setup") as mock_setup:
        import setup
    _, kwargs = mock_setup.call_args
    return kwargs


@pytest.mark.parametrize("file,key", [
    ("test_requirements.txt", "test"),
    ("doc_requirements.txt", "doc"),
], ids=["test_requirements", "doc_requirements"])
def test_requirements_txt(file: str, key: str, setup_kwargs: Dict[str, Any]) -> None:
    """Test that ``{x}_requirements.txt`` and ``setup.py`` files are synced."""
    with open(file, "r") as f:
        requirements_set = {PATTERN.sub("@", str(i)) for i in pkg_resources.parse_requirements(f)}

    setup_set = set(setup_kwargs["extras_require"][key])
    if key == "test":
        setup_set.update(setup_kwargs["install_requires"])

    diff_set = requirements_set ^ setup_set
    assertion.not_(diff_set, message=f"Dependency mismatch between setup.py and {file}")
