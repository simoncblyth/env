"""

   issues with Run :
 
      1)  with shell=True, the subprocess seems not to run ... just reaches timeout 
      
      2)  when timeout is reached, all output is lost 
           eg with 
                python run.py "python count.py 10" 1

          resolved this via readline... in the poll loop

      3) the output comes all at the end
                python run.py "python count.py 10" 10

"""

import subprocess, datetime, os, time, signal

class Run:
    def __init__(self, command , parser=None ):
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
        self.parser = parser

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
            time.sleep(0.2)
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
                    self.parser(output)
                if self.verbose: print "continuing process %s for %s " % ( process.pid , self )
        self.rc = process.returncode

    def __repr__(self):
        return "<Run \"%s\" rc:%s timeout:%s dur:%s >" % ( self.command , self.rc, self.timeout, self.dur )


    def __call__(self, timeout=100 , verbose=True  ):
        """
                  shell=True    for shell expansion of variables like $HOME in the cmdline
                    but seems to prevent running 

        """
        self.timeout = timeout
        self.verbose = verbose 
        self.start = datetime.datetime.now()
        cmd = self.command.split(" ")
        
        self.process = process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False )
        if self.verbose: print "forked process %s for %s  " % ( process.pid , self   )
        self.read()
        if self.verbose: print " completed  %s " % self




if __name__=='__main__':
    """
        python run.py "cat run.py" 10 
 
    """
    import sys
    run = Run(sys.argv[1]) 
    p = run(timeout=int(sys.argv[2]))
    #print p.stdout.readlines()


