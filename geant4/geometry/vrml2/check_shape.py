#!/usr/bin/env python
"""
"""

from shape import Shape
from shapecnf import parse_args


if __name__ == '__main__':
    opts, args = parse_args(__doc__)
    sh = Shape(100, opts)
    print sh 

