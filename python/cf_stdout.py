#!/usr/bin/env python 

"""
   NB see the object wrapped 
      version of this ... in cfref.py

"""

path = __file__ + ".ref"
from cStringIO import StringIO
cur = StringIO()
import sys


sys.stdout = cur    ## start capturing 

print "hello world "
from datetime import datetime
print datetime.now()


sys.stdout = sys.__stdout__  ## reset stdout 
cur_l = ["%s\n" % l  for l in cur.getvalue().split("\n")]

import os
if os.path.exists(path):
    ref = file(path,"r")
    ref_l = ref.readlines()
    from difflib import unified_diff
    print "comparing cur to %s " % path

    print "ref [%s] " % ref_l
    print "cur [%s] " % cur_l

    df = unified_diff( ref_l , cur_l  )
    ldf = list(df)
    if len(ldf) == 0 :
        print "matched "
    else:
        print "mismatch ... "
        for l in ldf:
            print l, 
else:
    print "no reference ... writing cur to %s " % path
    ref = file(path,"w")
    ref.writelines( cur_l )
    ref.close()
         


