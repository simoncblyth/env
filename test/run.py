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
      
      3)  too slow ...
          try re-use of CommandLine from  
             http://bitten.edgewall.org/browser/branches/experimental/trac-0.11/bitten/build/api.py
            
            
                  
    
"""

import subprocess, datetime, os, time, signal

def pp(d):
    import pprint
    return pprint.pformat(d)


class Run:
    def __init__(self, command , parser=None ,  opts=None ):
        """
           http://code.pui.ch/2007/02/19/set-timeout-for-a-shell-command-in-python/

           call shell-command and either return its output or kill it
           if it doesn't normally exit within timeout seconds and return None
        """
        self.command = command
        self.rc = None
        self.start = None
        self.dur = None
        self.parser = parser
        self.prc = 0
        self.opts = opts

    def assert_(self):
        assert self.rc  == 0 , self
        assert self.prc == 0 , self
        return self

    def overtime(self):
        self.dur =  (datetime.datetime.now() - self.start).seconds
        return self.dur - self.opts['timeout']

    def parse(self, out ):
        if out==None: return
        if self.parser==None:
            print out
        else:
            prc = self.parser(out)
            self.prc = max( self.prc, prc )


"""
    def read_1(self):
        process = self.process
        alive = True
        while alive:
            out = select.select...
"""        

    def read(self):
        """
            documentation suggests that process.returncode should be -signum for a signalled process
            but seems not to be so ... getting None
        """
        process = self.process
        while process.poll() is None:
            if self.overtime() > 0:
                self.rc = -666
                if self.opts['verbose']: print "timeout exceeded, killing process %s for %s " % ( process.pid , self )
                os.kill(process.pid, signal.SIGKILL)
                os.waitpid(-1, os.WNOHANG)
                #self.rc = process.returncode  
                return process
            else:
                out = process.stdout.readline()
                self.parse(out)
               #if self.verbose: print "continuing process %s for %s " % ( process.pid , self )
            time.sleep(0.2)
        self.rc = process.returncode

    def run_slow(self):
        cmd = self.command.split(" ")
        self.start = datetime.datetime.now()
        self.process = process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False )
        if self.opts['verbose']: print "forked process %s for %s  " % ( process.pid , self   )
        self.read()

    def run_fast(self):
        """
            "timeout" for select.select is not the same as my  "time to live"  ...
                http://docs.python.org/lib/module-select.html
                
                 0.  indicates a poll, never blocking 
                >0.  when timeout in seconds is reached with nothing ready return empties
               None  blocks until ready ... sneak this in by supplying a -ve 
             
            CommandLine doesnt provide access to the pipe... so cannot implement a 
            time-to-live like timeout 
             
        """
        from bitten.build import CommandLine
        cmd = self.command.split(" ")
        self.start = datetime.datetime.now()
        cmdline = CommandLine( cmd[0] , cmd[1:]  )
        if self.opts['verbose']: print "forked commandline %s   " % ( cmdline )
        
        sto = self.opts['select_timeout']
        if sto<0.:
            sto = None
        for out, err in cmdline.execute(timeout=sto):
            self.parse(out)
            self.parse(err)
            if self.opts['verbose']: print " continuing  %s " % cmdline
        self.dur =  (datetime.datetime.now() - self.start).seconds
        self.rc = cmdline.returncode
        if self.opts['verbose']: print " completed  %s " % cmdline

    def __repr__(self):
        return "<Run \"%s\" opts:%s prc:%s rc:%s  dur:%s parser:%s  >" % ( self.command , pp(self.opts), self.prc, self.rc , self.dur , self.parser )

    def __call__(self):
        if self.opts['slow']==True:
            self.run_slow()
        else:
            self.run_fast()
        return self


if __name__=='__main__':
    import sys
    r = Run( sys.argv[1] )().assert_()
   

