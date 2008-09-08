
import datetime
import subprocess
import select
import os
import time

class TimeoutError(Exception):
    pass

class CommandLine:
    """
        Simplification of CommandLine from  bitten/build/api.py 
    """
    def __init__(self , command  ):
        self.command = command
        self.duration = None
        self.killed = False
        self.returncode = None
        
    def __repr__(self):
        kmsg = ""
        if self.killed: kmsg = "KILLED" 
        return "<CommandLine \"%s\" %s %s %s  >" % (self.command, self.returncode , self.duration , kmsg )
        
    def __call__(self, **attr ):
        self.execute(**attr)
        return self
          
    def execute(self, bufsize=1024 , timeout=None , maxtime=None ):
        """ 
             Use of "stderr=subprocess.STDOUT" means that stderr is merged into stdout 
             
        select.select :
            When the timeout argument is omitted (or None) the function blocks until at least one file descriptor is ready. 
            A time-out value of zero specifies a poll and never blocks. The return value is a triple of lists
            of objects that are ready: subsets of the first three arguments. 
            When the time-out is reached without a file descriptor becoming ready, three empty lists are returned. 
             
        """
        self.bufsize = bufsize
        self.timeout = timeout 
        self.maxtime = maxtime
        cmd = self.command.split(" ")
        self.start = datetime.datetime.now()
        self.process = process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT , shell=False )
      
        readable = [ process.stdout ]    ## no process.stderr, due to the merging 
        writable = []
        xceptable = []
        
        out = []
        
        while readable: 
            read_ready,  wr, xr = select.select( readable , writable , xceptable , self.timeout )
            if not read_ready:
                raise TimeoutError("timeout for %s ", self )
            if process.stdout in read_ready:
                data = os.read( process.stdout.fileno() , self.bufsize )
                if data:
                    out.append(data)
                else:
                    readable.remove(process.stdout)
            if self.maxtime:
                self.duration =  (datetime.datetime.now() - self.start).seconds
                if self.duration > self.maxtime:
                    print "over allowed maxtime killing " 
                    self.kill()
            lines = self._extract_lines( out )
            for line in lines:
                yield line, None
            #time.sleep(.1)   ## sleeping can cause speedup, if avoids short reads
        
        if self.killed:
            self.returncode = -1
        else:
            self.returncode = process.wait()   ## block until completion
        print "subprocess returned ... %s " % self  
    
    def _extract_lines(self, data):
        """
            handles reads that do not end on line boundaries 
            by extracting only lines that have endings, setting the 
            data array to contain the leftover partial line, 
            ready for appending in the caller and extraction 
            into lines the next time around
        """
        extracted = []
        def _endswith_linesep(string):
            for linesep in ('\n', '\r\n', '\r'):
                if string.endswith(linesep):
                    return True
        buf = ''.join(data)
        lines = buf.splitlines(True)   ## keep endings
        if len(lines) > 1:
            extracted += lines[:-1]
            if _endswith_linesep(lines[-1]):
                extracted.append(lines[-1])
                buf = ''
            else:
                buf = lines[-1]
        elif _endswith_linesep(buf):
            extracted.append(buf)
            buf = ''
        data[:] = [buf] * bool(buf)   ## crucial clearing buffer or just setting the leftover partial line

        return [line.rstrip() for line in extracted]
    
    def kill(self):
        import signal, os
        if not(self.process):
            return
        pid = self.process.pid
        
        try:
            os.kill( pid, signal.SIGKILL)
            os.waitpid(-1, os.WNOHANG)
            self.killed = True
        except OSError:
            pass    ## it might complete before get here 

                                  
                
if __name__=='__main__':
    cl = CommandLine("python count.py 10")()
    