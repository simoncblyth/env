#!/usr/bin/env python
"""
Writes to log (stderr) every n seconds, where n is
one of several delays. To change the delay send SIGHUP
to the process.

When started under supervisord which uses full path can signal with::

        pkill -HUP -f python\ $ENV_HOME/tools/sv/demo_logger.py    
                ## -f matches the full command line

"""
import time, logging, signal, sys
log = logging.getLogger(__name__)

delays = [3,10,30,60]
index = 0    

def sighup_handler(signal, frame):
    global index
    index = (index + 1) % len(delays)
    log.info("received signal %s move to index %s " % (signal,index) )

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s")
    while 1:
        delay = delays[index]
        log.info("hi using index %s delay %s " % (index, delay))
        time.sleep(delay)

signal.signal(signal.SIGHUP, sighup_handler)

if __name__ == '__main__':
    main()




