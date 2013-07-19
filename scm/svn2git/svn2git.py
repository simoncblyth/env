#!/usr/bin/env python
"""
"""
import os, logging
from config import parse_args
from svnauthors import Authors
log = logging.getLogger(__name__)


if __name__ == '__main__':
    cnf, args = parse_args(__doc__)
    au = Authors(cnf)
    gitauth = au.git()

    print gitauth


