#!/usr/bin/env python
"""
http://pymotw.com/2/socket/udp.html
"""
import logging
import os
import socket
import sys

log = logging.getLogger(__name__)

def send_recv(sock, msg, host, port):
    log.info("send_recv [%s] to host:port %s:%s " % (msg, host, port))
    try:
        sent = sock.sendto(msg, (host,int(port)))
        data, server = sock.recvfrom(4096)
        log.info("revcfrom [%s] [%s] " % (server, data) )
    finally:
        sock.close()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    msg = " ".join(sys.argv[1:])
    host = os.environ.get("UDP_HOST","127.0.0.1")
    port = os.environ.get("UDP_PORT","8080")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_recv(sock, msg, host, port )


if __name__ == '__main__':
    main()




