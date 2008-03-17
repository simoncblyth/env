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

def makefile(MB=100):
	''' create a file of requested number of  megabytes returning the name of the file '''
	f="%sm" % MB 
	name="makefile%s" % f

	if sys.platform == "darwin" :
		mmd = "mkfile %s %s" % ( f, name)
	elif sys.platform == "linux2" :
		mmd = "dd if=/dev/zero of=%s bs=1024 count=%d " % ( name , 1024*MB )
	else:
		mmd = "echo sorry platform %s not supported... cannot create test file  " % sys.platform

	if not os.path.exists(name):
		print "print creating testfile with command: %s " % mmd  
		os.system(mmd)
	return name 

def timed(cmd):
	t0=time.time()
	os.system( cmd )
	t1=time.time()
	return t1-t0 

def present_mb( n, t):
	MBps = MB / t 
	mbps = MBps * 8  
	return "time to copy %d MB is  %3.2f s  which corresponds to:  %3.2f MB/s  or %3.2f mbps " % ( MB , t , MBps , mbps  )

def present_test( n , cmd ):
	t = timed( cmd )
	return " === %s === %s " % ( cmd , present_mb( n , t ) ) 


if __name__=="__main__":
	
	if len(sys.argv) > 1 and sys.argv[1].startswith("/"):
		todir=sys.argv[1]
	else:
		todir="/tmp"
	
	os.chdir("/tmp")
	MB=100
	name=makefile( MB )      ## make a 100MB local file in PWD

	na=name
	nb="%s.1" % name
	
	fa=na
	fb="%s/%s" % ( todir , nb )
	
	print present_test( MB , "cp -f %s %s " % ( fa , fb ) )
	print present_test( MB , "cp -f %s %s " % ( fb , fa ) )



