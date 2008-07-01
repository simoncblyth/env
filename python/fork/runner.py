"""
    DEVELOPMENT ABANDONED DUE TO DISCOVERY OF "InsulateRunner" NOSE PLUGIN

   based on /Users/blyth/roadrunner-0.2.2/roadrunner/runner.py 
   allows tests to be run in a separate process avoiding the 
   need for stringent cleanup and preventing test failures that 
   cause crashes from killing the rest if the tests  
   
   when the child exits due to payload failure the bad behaviour of 
   recursive forking and overreporting happens again 
   ... need nose to catch the error but not to go so crazy over it 
   
"""
import os, sys, time, signal

def simple_runner(*args,**kwargs):
    assert callable(args[0])
    return args[0](args[1:])

original_signal_handler = None

def register_signal_handlers(pid):
    "propogate signals to child process"
    def interrupt_handler(signum, frame, pid=pid):
        try:
            print "interrupt_handler invoked pid %d " % pid 
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

def log(msg):
    print "\n ppid/pid %s/%s msg %s " % ( os.getppid(), os.getpid() , msg )

def forking_runner(*args,**kwargs):
    assert callable(args[0])
    log("enter forking runner before fork")
    pid = os.fork()
    if not pid:
        log("child start ")
        t1=time.time()
        rc = args[0](args[1:])
        log("child sleeping")
        time.sleep(1)
        t2=time.time()
        log("child runner took : %0.3f seconds rc: %d " % ((t2-t1),rc))
        
        # exit the child quietly to prevent nose over-reporting 
        # and bizarre recursive forking when had sys.exit(rc) here
        os._exit(rc)
    else:
        log("parent")
        try:
            register_signal_handlers(pid)
            try:
                #status = os.wait()
                status = os.waitpid(pid,0)
                log("parent> wait returned %s " % repr(status))
            except OSError:
                log("OSError noted in parent ")
                pass
        finally:
            log("parent finally")
            ignore_signal_handlers()
            os.system("stty echo") # HACK
    log("exit forking runner")



