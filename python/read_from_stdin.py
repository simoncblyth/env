#!/usr/bin/env python
"""

::

    delta:env blyth$ cat python/read_from_stdin.py | python/read_from_stdin.py
    ['#!/usr/bin/env python\n', '\n', 'import sys\n', '\n', 'print sys.stdin.readlines()\n']

"""
import sys

lines = map(str.strip, sys.stdin.readlines() )

print lines
