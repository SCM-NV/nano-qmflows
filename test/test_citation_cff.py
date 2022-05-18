from datetime import datetime
from pathlib import Path

import yaml
from packaging.version import Version
from assertionlib import assertion

import nanoqm

CITATION_FILE = Path(__file__).parents[1] / "CITATION.cff"
with open(CITATION_FILE, "r", encoding="utf8") as f:
    CITATION_DCT = yaml.load(f.read(), Loader=yaml.SafeLoader)


def test_date(is_release: bool) -> None:
    date = datetime.strptime(CITATION_DCT["date-released"], "%Y-%m-%d").date()
    today = datetime.today().date()
    if is_release:
        assertion.eq(date, today, message="CITATION.cff date-released mismatch")


def test_version(is_release: bool) -> None:
    version = Version(CITATION_DCT["version"])
    nanoqm_version = Version(nanoqm.__version__)
    assertion.not_(version.is_devrelease, message="CITATION.cff version dev release")
    if is_release:
        assertion.eq(version, nanoqm_version, message="CITATION.cff version mismatch")
