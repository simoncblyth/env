#!/usr/bin/env python
import sys
import zmq
import logging
log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    context = zmq.Context()
    sock = context.socket(zmq.REP)
    endpoint = "tcp://*:%s" % sys.argv[1]
    log.info("server binding to REP endpoint %s " % endpoint )
    sock.bind(endpoint)

    while True:
        x = sock.recv()
        print 'REQ is', x,
        reply = 'x-%s' % x
        sock.send(reply)
        print 'REP is', reply


if __name__ == '__main__':
    main()


