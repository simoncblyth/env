#!/usr/bin/env python
"""
http://pymotw.com/2/socket/udp.html
"""
import logging
import os
import socket
import select
import sys
import argparse


log = logging.getLogger(__name__)

def send(sock, args):
    log.info("send [%s] to %s " % (args.msg, args.addr))
    try:
        sent = sock.sendto(args.msg, args.addr)
    finally:
        sock.close()
    pass

def sendrecv(sock, args):
    log.info("sendrecv [%s] to %s " % (args.msg, args.addr))
    try:
        sent = sock.sendto(args.msg, args.addr)
        while True:
            data, server = sock.recvfrom(args.bufsize)
            log.info("revcfrom [%s] [%s] " % (server, data) )
    finally:
        sock.close()
    pass

def recvfrom(sock, args):
    """
    https://wiki.python.org/moin/UdpCommunication
    Works but crashes with EBADF bad file descriptors
    """
    log.info("recv from %s " % str(args.addr) )
    bind(sock, args)
    while 1:
        try:
            data, server = sock.recvfrom(args.bufsize)
            log.info("revcfrom [%s] [%s] " % (server, data) )
        finally:
            sock.close()
        pass
    pass

def recv(sock, args):
    log.info("recv/select %s " % str(args.addr) )
    bind(sock, args)
    sock.setblocking(0)
    while True:
        result = select.select([sock],[],[])
        msg = result[0][0].recv(args.bufsize) 
        log.info(msg)
    pass

def bind(sock, args):
    if args.bind:
        log.info("binding to %s" % str(args.addr))
        sock.bind(args.addr)
    else:
        log.info("not binding")
    pass


def parse_args(doc):
    parser = argparse.ArgumentParser(doc)
    d = {}

    d['level'] = "INFO"
    d['msg'] = "hello world"
    d['bind'] = True 
    parser.add_argument("--level", default=d['level'] ) 
    parser.add_argument("msg", nargs='*', default=d['msg'] )
    parser.add_argument("-B", "--nobind", action="store_false", dest="bind", default=d['bind'] ) 

    d['send'] = False 
    d['sendrecv'] = False 
    d['recv'] = False
    d['recvfrom'] = False
    parser.add_argument("--send",     action="store_true", default=d['send'] ) 
    parser.add_argument("--sendrecv", action="store_true", default=d['sendrecv'] ) 
    parser.add_argument("--recv",     action="store_true", default=d['recv'] ) 
    parser.add_argument("--recvfrom", action="store_true", default=d['recvfrom'] ) 

    d['host'] = os.environ.get("UDP_HOST","127.0.0.1")
    d['port'] = os.environ.get("UDP_PORT","8080")
    d['bufsize'] = 4096
    parser.add_argument("--host", default=d['host'] ) 
    parser.add_argument("--port", type=int, default=d['port'] ) 
    parser.add_argument("--bufsize", type=int, default=d['bufsize'] ) 

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.level.upper()),format="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    args.msg = " ".join(map(str,args.msg))
    args.addr = (args.host, args.port)
    return args 



def main():
    args = parse_args(__doc__)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)     # UDP

    if args.sendrecv: 
        sendrecv(sock, args )
    elif args.recv:
        recv(sock, args )
    elif args.recvfrom:
        recvfrom(sock, args )
    elif args.send:
        send(sock, args )
    else:
        log.info("use one of : --sendrecv/--recv/--recvfrom/--send  to do something")
    pass

    sock.close()
  


if __name__ == '__main__':
    main()




