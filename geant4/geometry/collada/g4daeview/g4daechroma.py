#!/usr/bin/env python
"""
Plain ZMQRoot responder and Chroma propagator with no OpenGL/glumpy   
"""
import logging, time
log = logging.getLogger(__name__)

from daedirectconfig import DAEDirectConfig
from daedirectresponder import DAEDirectResponder

def main():
    print "main"
    config = DAEDirectConfig(__doc__)
    config.parse()
    responder = DAEDirectResponder( config )
    log.info("polling: %s " % repr(responder))
    count = 0 
    while True:
        log.info("polling %s " % count ) 
        responder.poll()
        count += 1 
        time.sleep(1) 
    pass
    log.info("terminating")


if __name__ == '__main__':
    main()




