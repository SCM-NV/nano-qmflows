"""Test the analysis tools."""

from nanoqm.analysis.tools import parse_list_of_lists


def test_list_parser():
    """Check that a string representing a list of list is parsed correctly."""
    xs = '[[1,2,3,4]]'
    result = parse_list_of_lists(xs)
    assert result[0] == [1, 2, 3, 4]
