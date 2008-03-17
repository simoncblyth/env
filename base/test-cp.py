#!/usr/bin/env python
""" 
    measureing the time to copy files 
        ... eg 100MB for a reasonable measurement

    verified to be consistent with the "real" results from command line: 
        time  cp 100m 100m.1  

   OOPS mkfile seems to be an OS X thing 

     usage :
          cd ~/env/base
          python test-cp.py

 """

import os
import time
import sys


print sys.argv


## filesize in megabytes
MB=100
f="%sm" % MB 

if sys.platform == "darwin" :
    mmd = "mkfile %s %s" % ( f, f)
elif sys.platform == "linux2" :
    mmd = "dd if=/dev/zero of=%s bs=1024 count=%d " % ( f , 1024*MB )
else:
    mmd = "echo sorry platform %s not supported... cannot create test file  " % sys.platform

os.chdir("/tmp")
if not os.path.exists(f):
    print "print creating testfile with command: %s " % mmd  
    os.system(mmd)


p = "%s.%s" % ( f, "1" )
if os.path.exists( p ):
    os.system("rm -f %s " % p )

if sys.argv[1].startswith("/"):
	todir=sys.argv[1]
else:
	todir="/tmp"

cmd = "cp %s %s/%s " % ( f, todir ,  p )

print " timing command %s " % cmd 
t0=time.time()
os.system( cmd )
t1=time.time()

t = t1-t0 
MBps = MB / t 
mbps = MBps * 8  

print "time to copy %d MB is  %3.2f s  which corresponds to:  %3.2f MB/s  or %3.2f mbps " % ( MB , t , MBps , mbps  )




