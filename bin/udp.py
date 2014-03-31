#!/usr/bin/env python
import os, sys, socket

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = os.environ.get("UDP_PORT","15006")
    ip = os.environ.get("UDP_IP","127.0.0.1")
    
    arg = sys.argv[1]
    print "sending [%s] to ip:port %s:%s " % (arg, ip, port) 
    sock.sendto(arg, (ip,int(port)))

if __name__ == '__main__':
    main()



