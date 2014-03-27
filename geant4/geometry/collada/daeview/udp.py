#!/usr/bin/env python
import sys, socket

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 15006
    ip = "127.0.0.1"
    arg = sys.argv[1]
    print "sending %s " % arg 
    sock.sendto(arg, (ip,port))

if __name__ == '__main__':
    main()



