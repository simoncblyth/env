#!/usr/bin/env python
"""
Plain ZMQRoot responder and Chroma propagator with no OpenGL/glumpy   
"""
import logging, time
log = logging.getLogger(__name__)

from daedirectconfig import DAEDirectConfig
from daedirectresponder import DAEDirectResponder

def main():
    config = DAEDirectConfig(__doc__)
    config.parse()
    responder = DAEDirectResponder( config )
    log.info("polling: %s " % repr(responder))
    responder.poll()
    log.info("sleeping while waiting for cpl to arrive")
    time.sleep(1000000) 
    log.info("terminating")


if __name__ == '__main__':
    main()




