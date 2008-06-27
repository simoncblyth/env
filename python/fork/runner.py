

import os, sys, time, signal

original_signal_handler = None

def register_signal_handlers(pid):
    "propogate signals to child process"
    def interrupt_handler(signum, frame, pid=pid):
        try:
            os.kill(pid, signal.SIGKILL)
            print # clear the line
        except OSError, e:
            print str(e), pid

def default_int_handler(signum, frame):
    print "\nInterrupt received. Type 'exit' to quit."
    
def ignore_signal_handlers():
    "restore signal handler"
    signal.signal(signal.SIGINT, default_int_handler)


def forking_runner(*args,**kwargs):
    """ based on   /Users/blyth/roadrunner-0.2.2/roadrunner/runner.py """
    assert callable(args[0])
    pid = os.fork()
    if not pid:
        ## child process
        t1=time.time()
        rc = args[0](args[1:])
        t2=time.time()
        print "child runner took : %0.3f seconds " % (t2-t1)
        sys.exit(rc)
    else:
        ## parent process
        try:
            register_signal_handlers(pid)
            try:
                status = os.wait()
            except OSError:
                print "OSError noted in parent "
                pass
        finally:
            ignore_signal_handlers()
            os.system("stty echo") # HACK
        
