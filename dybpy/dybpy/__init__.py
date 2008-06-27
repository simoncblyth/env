"""
  For quick and dirty usage :
     from dybpy import *
"""

NAME = 'dybpy'

import sys

def syspath():
    import sys
    for s in sys.path:
        print s

def my_reload():
    print " reloading %s " % NAME 
    reload(sys.modules[NAME])


from minimal import *

