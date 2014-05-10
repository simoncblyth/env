#!/usr/bin/env python
"""




"""
import sys
import zmq
import logging
log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    context = zmq.Context()

    s1 = context.socket(zmq.XREQ)
    s2 = context.socket(zmq.XREP)

    e1 = "tcp://localhost:%s" % sys.argv[1]
    e2 = "tcp://localhost:%s" % sys.argv[2]

    log.info("%s connecting to frontend XREQ endpoint %s and backend XREP endpoint %s " % (sys.argv[0], e1, e2))
    s1.connect(e1)
    s2.connect(e2)
    
    zmq.device(zmq.QUEUE, s1, s2)


if __name__ == '__main__':
    main()

