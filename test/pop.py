"""

  Better to use subprocess rather than popen2
    http://developer.spikesource.com/wiki/index.php/How_to_invoke_subprocesses_from_Python

     p = subprocess.Popen( "sleep 10000000".split(" ") )

In [8]: p.pid
Out[8]: 7495


  Module to allow Asynchronous subprocess use on Windows and Posix platforms
      http://code.activestate.com/recipes/440554/
    using fcntl and select 
   for the current application ... just want to monitor a running process, and parse the 
   output ... do need for this complication


"""
import string
import re

                
      
def chk_( command , checks={} , **atts  ):
    """

    """
    mtch = Matcher(checks)
    run = Run(command) 
    p = run(**atts)
    for line in p.stdout.readlines():
        code = mtch.match( line )
        print "[%-1s] %s " % ( code , line ),  


if __name__=='__main__':
#    cmd_("python share/geniotest.py")
#    cmd_("python share/geniotest.py input")

    import sys
    
    ptns = {  '.*FATAL':1 ,  '.*ERROR':2 } 
    chk_("cat %s" % sys.argv[0] , ptns ,  timeout=10 )

    chk_( "sleep 100", {} , timeout=1  )




