"""
    Runs a command in a subprocess :
      * if a timeout is exceeded the subprocess is killed 
        and a negative return code is set
      * merged stdout+stderr from the subprocess is handed over to a parser line-by-line 
        The parser yields return codes "prc" for each line, the maximum such 
        code is stored in the Run instance
      * an assert_ method allows nosetests to be setup with minimal code,
        see runner.py for examples  
          
     Example :
        python run.py "python count.py 10" 

    issues with Run :
 
      1)  with shell=True, the subprocess seems not to run 
          ... just reaches timeout 
      
"""

import subprocess, datetime, os, time, signal
from command import CommandLine
        
def pp(d):
    import pprint
    return pprint.pformat(d)


defaults = {  'maxtime':5 , 'verbose':True , 'timeout':-1. }

class Run:
    def __init__(self, command , parser=None ,  opts=None ):
        """
            Parsing runner performs the command in a subprocess, monitoring the 
            merged stdout + stderr with the parser     
        """
        self.cmdline = CommandLine( command )
        self.parser = parser
        self.prc = 0        
        self.opts = defaults
        if opts:
            self.opts.update( **opts )

    def assert_(self):
        self.cmdline.assert_()
        assert self.prc == 0 , self
        return self

    def parse(self, out ):
        if out==None: return
        if self.parser==None:
            print out
        else:
            prc = self.parser(out)
            self.prc = max( self.prc, prc )

    def run(self):
        cmdline = self.cmdline
        if self.opts['verbose']: print "forking %s " % ( self )
        timeout = self.opts['timeout']
        if timeout<0.:
            timeout = None
            
        for out, err in self.cmdline.execute(timeout=timeout, maxtime=self.opts['maxtime'] ):
            self.parse(out)
            self.parse(err)
            #if self.opts['verbose']: print " continuing  %s " % cmdline
       
        if self.opts['verbose']: print "completed %s " % ( self )

    def __repr__(self):
        return "<Run \"%s\" opts:%s prc:%s parser:%s  >" % ( self.cmdline , pp(self.opts), self.prc, self.parser )

    def __call__(self):
        self.run()
        return self


if __name__=='__main__':
    import sys
    r = Run( sys.argv[1] , opts={'slow':True , 'verbose':True , 'timeout':-1 , 'maxtime':10 } )().assert_()
   

