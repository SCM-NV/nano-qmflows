"""Check that the cp2k error is print to the user."""

from pathlib import Path

import pytest
from nanoqm.schedule.scheduleCP2K import try_to_read_wf


def test_cp2k_call_error(tmp_path: Path):
    """Check cp2k error files."""
    path_err = tmp_path / "cp2k.err"
    with open(path_err, 'w') as handler:
        handler.write("Some CP2K error")

    with pytest.raises(RuntimeError) as info:
        try_to_read_wf(tmp_path)

    error = info.value.args[0]
    print(error)
    assert "CP2K error" in error
