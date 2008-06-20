"""
  For quick and dirty usage :
     from dybpy import *
"""

NAME = 'dybpy'

import sys
def my_reload():
    print " reloading %s " % NAME 
    reload(sys.modules[NAME])


from geneventlook import *

