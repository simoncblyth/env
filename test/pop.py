"""

  Better to use subprocess rather than popen2
    http://developer.spikesource.com/wiki/index.php/How_to_invoke_subprocesses_from_Python

     p = subprocess.Popen( "sleep 10000000".split(" ") )

In [8]: p.pid
Out[8]: 7495


"""
import string
import re
import subprocess, datetime, os, time, signal

class Run:
    def __init__(self, command ):
        """
           http://code.pui.ch/2007/02/19/set-timeout-for-a-shell-command-in-python/

           call shell-command and either return its output or kill it
           if it doesn't normally exit within timeout seconds and return None
        """
        self.command = command
        self.rc = None
        self.timeout = None
        self.start = None
        self.dur = None

    def __call__(self, timeout=100, verbose=True ):
        cmd = self.command.split(" ")
        self.start = datetime.datetime.now()
        self.timeout = timeout 
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if verbose: print "forked process %s for %s  " % ( process.pid , self   )
        while process.poll() is None:
            time.sleep(0.2)
            self.dur =  (datetime.datetime.now() - self.start).seconds
            if self.dur > timeout:
                self.rc = -666
                if verbose: print "timeout exceeded, killing process %s for %s " % ( process.pid , self )
                os.kill(process.pid, signal.SIGKILL)
                os.waitpid(-1, os.WNOHANG)
                return process
        self.rc = process.returncode
        if verbose: print " completed  %s " % self
        return process 

    def __repr__(self):
        return "<Run \"%s\" rc:%s dur:%s timeout:%s start:%s  >" % ( self.command , self.rc , self.dur, self.timeout, self.start  ) 




class Matcher:
    def __init__(self, patterns ):
        self.patns={}
        for pt,rc in patterns.items():
            self.patns[pt] = re.compile(pt), rc  
    def match( self , line ):
        """ return the code of the first match, or zero if no match """
        for pt in self.patns.keys():
             if self.patns[pt][0].match(line)>-1:
                  return self.patns[pt][1]
        return 0     
                 
      
def chk_( command , timeout , checks ):
    """

    """
    mtch = Matcher(checks)

    run = Run(command) 
    p = run(timeout)
    for line in p.stdout.readlines():
        code = mtch.match( line )
        print "[%-1s] %s " % ( code , line ),  


if __name__=='__main__':
#    cmd_("python share/geniotest.py")
#    cmd_("python share/geniotest.py input")

    import sys
    
    ptns = {  '.*FATAL':1 ,  '.*ERROR':2 } 
    chk_("cat %s" % sys.argv[0] , 10 , ptns  )

    chk_( "sleep 100", 1 , {} )




