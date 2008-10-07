#!/usr/bin/env python 
"""
   Using trac + plugin facilities from the command line ... by use of 
   an EnvironmentStub and Mock request. 

   Good for fast turnaround testing of formatters ...

"""

import trac2latex
import sys
if len(sys.argv)>1:
    path = sys.argv[1]
    source = file(path).read()
else:
    source = "".join(sys.stdin.readlines())

trac2latex.cli_convert( source )



