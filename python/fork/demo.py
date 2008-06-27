"""
   Python Forking investigations, following the pattern of 
       /Users/blyth/roadrunner-0.2.2/roadrunner/runner.py
"""

import os, sys, time, signal
from configiter import configiter

original_signal_handler = None

def register_signal_handlers(pid):
    "propogate signals to child process"
    def interrupt_handler(signum, frame, pid=pid):
        try:
            os.kill(pid, signal.SIGKILL)
            print # clear the line
        except OSError, e:
            print str(e), pid

    signal.signal(signal.SIGINT, interrupt_handler)

def default_int_handler(signum, frame):
    print "\nInterrupt received. Type 'exit' to quit."
    
def ignore_signal_handlers():
    "restore signal handler"
    signal.signal(signal.SIGINT, default_int_handler)

def forking_loop(citer):

    ignore_signal_handlers()
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    for ctpl in citer:
        # Test Loop Start
        pid = os.fork()
        if not pid:
            # Run tests in child process
            t1 = time.time()
            rc = runner(ctpl)
            t2 = time.time()
            print 'child runner took: %0.3f seconds.  ' % ((t2-t1))
            sys.exit(rc)

        else:
            # In parent process
            try:
                register_signal_handlers(pid)
                try:
                    # os.wait() can throw OSError
                    status = os.wait()
                except OSError:
                    # TODO: maybe need something smarter here, maybe not?
                    pass
            finally:
                ignore_signal_handlers()
                # TODO: deal with windows
                os.system("stty echo") # HACK
                
            args = run_commandloop(args)

    # start the test loop over....



if __name__=='__main__':
    

