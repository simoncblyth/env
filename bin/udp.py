#!/usr/bin/env python
import os, sys, socket

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = os.environ.get("UDP_PORT","15006")
    host = os.environ.get("UDP_HOST","127.0.0.1")
    
    arg = " ".join(sys.argv[1:])
    print "sending [%s] to host:port %s:%s " % (arg, host, port) 
    sock.sendto(arg, (host,int(port)))

if __name__ == '__main__':
    main()



