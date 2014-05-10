#!/usr/bin/env python
"""
Need to connect machines that cannot directly see each other
because one or both are on private networks.

http://blog.pythonisito.com/2012/08/using-zeromq-devices-to-support-complex.html

"""
import logging
import sys
import zmq


def main():
    context = zmq.Context()

    s1 = context.socket(zmq.ROUTER)
    s2 = context.socket(zmq.DEALER)

    s1.bind(sys.argv[1])
    s2.bind(sys.argv[2])

    zmq.device(zmq.QUEUE, s1, s2)


if __name__ == '__main__':
    main()

