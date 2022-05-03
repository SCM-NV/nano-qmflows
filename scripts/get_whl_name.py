"""Find the first file in the passed directory matching a given regex pattern."""

from __future__ import annotations

import os
import re
import argparse


def main(directory: str | os.PathLike[str], pattern: str | re.Pattern[str]) -> str:
    pattern = re.compile(pattern)
    for i in os.listdir(directory):
        if pattern.search(i) is not None:
            return os.path.join(directory, i)
    else:
        raise FileNotFoundError(
            f"Failed to identify a file in {os.fspath(directory)!r} "
            f"with the following pattern: {pattern!r}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python get_whl_name.py . manylinux2014_x86_64", description=__doc__
    )
    parser.add_argument("directory", help="The to-be searched directory")
    parser.add_argument("pattern", help="The to-be searched regex pattern")

    args = parser.parse_args()
    print(main(args.directory, args.pattern))
