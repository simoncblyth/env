#!/usr/bin/env python
import sys
import zmq
import logging
log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    context = zmq.Context()
    sock = context.socket(zmq.REQ)

    endpoint = "tcp://localhost:%s" % sys.argv[1]
    log.info("%s connecting REQ to endpoint %s " % (sys.argv[0], endpoint))
    sock.connect(endpoint)

    for x in xrange(10):
        print 'REQ is', x,
        sock.send(str(x))
        print 'REP is', sock.recv()

if __name__ == '__main__':
    main()



