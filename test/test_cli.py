"""Test the run_runworkflow script."""

import argparse
from pathlib import Path

import pytest
import yaml
from nanoqm.common import UniqueSafeLoader
from nanoqm.workflows.run_workflow import main
from pytest_mock import MockFixture

from .utilsTest import PATH_TEST


def call_main(mocker: MockFixture, path_input: Path, scratch_path: Path):
    """Mock main function."""
    # Mock argparse
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        i=path_input))

    mocker.patch("nanoqm.workflows.run_workflow.process_input", return_value={})
    mocker.patch("nanoqm.workflows.run_workflow.dict_workflows", return_value=len)
    main()


def test_run_workflow(mocker: MockFixture, tmp_path: Path):
    """Test that the CLI main command is called correctly."""
    path_input = PATH_TEST / "input_fast_test_derivative_couplings.yml"
    call_main(mocker, path_input, tmp_path)


def test_run_workflow_no_workflow(mocker: MockFixture, tmp_path: Path):
    """Check that an error is raised if not workflow is provided."""
    # remove workflow keyword
    with open(PATH_TEST / "input_fast_test_derivative_couplings.yml", 'r') as handler:
        input = yaml.load(handler, UniqueSafeLoader)
    input.pop('workflow')
    path_input = tmp_path / "wrong_input.yml"
    with open(path_input, 'w') as handler:
        yaml.dump(input, handler)

    with pytest.raises(RuntimeError) as info:
        call_main(mocker, path_input, tmp_path)

    error = info.value.args[0]
    assert "is required" in error
