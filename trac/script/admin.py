

import sys
import os

from trac.admin.console import TracAdmin
from trac.ticket.model import *

def run(args):
    envdir = args[0]
    admin = TracAdmin()
    admin.env_set(envdir) 
    #admin.onecmd("component list")
    for c in Component.select(admin.env_open()):
        print "c.name [%s] c.owner [%s] " % (c.name,c.owner)

if __name__=='__main__':
    run(sys.argv[1:])


