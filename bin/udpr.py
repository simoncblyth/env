#!/usr/bin/env python
"""
http://pymotw.com/2/socket/udp.html
"""
import logging
import os
import socket
import sys
import argparse


log = logging.getLogger(__name__)

def send_recv(sock, msg, host, port):
    log.info("send_recv [%s] to host:port %s:%s " % (msg, host, port))
    try:
        sent = sock.sendto(msg, (host,int(port)))
        data, server = sock.recvfrom(4096)
        log.info("revcfrom [%s] [%s] " % (server, data) )
    finally:
        sock.close()

def recv(sock, host, port):
    log.info("recv from host:port %s:%s " % (host, port))
    while 1:
        try:
            data, server = sock.recvfrom(4096)
            log.info("revcfrom [%s] [%s] " % (server, data) )
        finally:
            sock.close()


def main():
    parser = argparse.ArgumentParser()
    d = {}
    d['msg'] = "hello world"
    d['recv'] = False
    d['level'] = "INFO"
    d['host'] = os.environ.get("UDP_HOST","127.0.0.1")
    d['port'] = os.environ.get("UDP_PORT","8080")

    parser.add_argument("msg", nargs='*', default=d['msg'] )
    parser.add_argument("--recv", help="only receive", action="store_true", default=d['recv'] ) 
    parser.add_argument("--host", default=d['host'] ) 
    parser.add_argument("--port", default=d['port'] ) 
    parser.add_argument("--level", default=d['level'] ) 

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.level.upper()),format="%(asctime)s %(name)s %(levelname)-8s %(message)s" )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if args.recv: 
        recv(sock, args.host, args.port )
    else:
        send_recv(sock, " ".join(map(str,args.msg)), args.host, args.port )


if __name__ == '__main__':
    main()




