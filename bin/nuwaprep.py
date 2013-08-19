#!/usr/bin/env python
"""
Prepares an time stamped output directory 
structure for a nuwa.py run within the invoking directory, eg::

    [blyth@belle7 tmp]$ find 20130815-2030/
    20130815-2030/
    20130815-2030/out
    20130815-2030/log

And emits the absolute path to the directory on stdout
"""
import sys
import os
from datetime import datetime

prepdir = lambda dir:os.path.exists(dir) or os.makedirs(dir)
stamp = datetime.now().strftime("%Y%m%d-%H%M")
prepdir(stamp)
os.chdir(stamp)
map(prepdir, sys.argv[1:])

print os.path.abspath(os.curdir)


