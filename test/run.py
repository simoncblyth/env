"""
    Runs a command in a subprocess :
      * if a timeout is exceeded the subprocess is killed 
        and a negative return code is set
      * stdout from the subprocess is handed over to a parser line-by-line 
        The parser yields return codes "prc" for each line, the maximum such 
        code is stored in the Run instance
      * an assert_ method allows nosetests to be setup with minimal code,
        see runner.py for examples  
          
          
     Example :
        python run.py "python count.py 10" 1

    issues with Run :
 
      1)  with shell=True, the subprocess seems not to run 
          ... just reaches timeout 
      
      2)  stderr is currently being ignored ...
         
    
"""

import subprocess, datetime, os, time, signal

class Run:
    def __init__(self, command , parser=None , timeout=300, verbose=True):
        """
           http://code.pui.ch/2007/02/19/set-timeout-for-a-shell-command-in-python/

           call shell-command and either return its output or kill it
           if it doesn't normally exit within timeout seconds and return None
        """
        self.command = command
        self.rc = None
        self.timeout = timeout
        self.start = None
        self.dur = None
        self.parser = parser
        self.prc = 0
        self.verbose = verbose 

    def assert_(self):
        assert self.rc  == 0 , self
        assert self.prc == 0 , self
        return self

    def overtime(self):
        self.dur =  (datetime.datetime.now() - self.start).seconds
        return self.dur - self.timeout

    def read(self):
        """
            documentation suggests that process.returncode should be -signum for a signalled process
            but seems not to be so ... getting None
        """
        process = self.process
        while process.poll() is None:
            if self.overtime() > 0:
                self.rc = -666
                if self.verbose: print "timeout exceeded, killing process %s for %s " % ( process.pid , self )
                os.kill(process.pid, signal.SIGKILL)
                os.waitpid(-1, os.WNOHANG)
                #self.rc = process.returncode  
                return process
            else:
                output = process.stdout.readline()
                if self.parser==None:
                    print output
                else:
                    prc = self.parser(output)
                    self.prc = max( self.prc, prc )
                #if self.verbose: print "continuing process %s for %s " % ( process.pid , self )
            time.sleep(0.2)
        self.rc = process.returncode

    def __repr__(self):
        return "<Run \"%s\" prc:%s rc:%s timeout:%s dur:%s parser:%s  >" % ( self.command , self.prc, self.rc, self.timeout, self.dur , self.parser )

    def __call__(self):
        self.start = datetime.datetime.now()
        cmd = self.command.split(" ")
        
        self.process = process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False )
        if self.verbose: print "forked process %s for %s  " % ( process.pid , self   )
        self.read()
        if self.verbose: print " completed  %s " % self
        return self

if __name__=='__main__':
    import sys
    r = Run(sys.argv[1], timeout=int(sys.argv[2]) )().assert_()
   

