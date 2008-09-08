

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


    def overtime(self):
        self.dur =  (datetime.datetime.now() - self.start).seconds
        return self.dur - self.opts['timeout']
