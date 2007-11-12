#!/usr/bin/env python
""" measureing the time to copy files ... eg 100MB for a reasonable measurement """

import os
import time


## filesize in megabytes
MB=100
f="%sm" % MB 

os.chdir("/tmp")
if not os.path.exists(f):
    print "print creating testfile %s " % f  
    os.system( "mkfile %s %s" % ( f, f))

p = "%s.%s" % ( f, "1" )
if os.path.exists( p ):
    os.system("rm -f %s " % p )


cmd = "cp %s %s " % ( f, p )

print " timing command %s " % cmd 
t0=time.time()
os.system( cmd )
t1=time.time()

t = t1-t0 
MBps = MB / t 
mbps = MBps * 8  

print "time to copy %d MB is  %3.2f s  which corresponds to:  %3.2f MB/s  or %3.2f mbps " % ( MB , t , MBps , mbps  )




