#!/usr/bin/env python

import os

rstpath = "%s%s" % ( os.path.splitext(os.path.abspath(__file__))[0], ".rst")
print rstpath

for _ in open(rstpath, "r").readlines():
    print _,
    print str(_),




