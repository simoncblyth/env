#!/usr/bin/env python
import os, sys, socket
"""
https://wiki.python.org/moin/UdpCommunication

http://pymotw.com/2/socket/udp.html

UDP messages must fit within a single packet (for IPv4, that means they can
only hold 65,507 bytes because the 65,535 byte packet also includes header
information) and delivery is not guaranteed as it is with TCP.

"""

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = os.environ.get("UDP_PORT","15001")   # was 15006
    host = os.environ.get("UDP_HOST","127.0.0.1")
    
    arg = " ".join(sys.argv[1:])
    print("sending [%s] to host:port %s:%s " % (arg, host, port))
    sock.sendto(arg.encode(), (host,int(port)))

    #data, server = sock.recvfrom(4096)
    #print("received %s " % data)



if __name__ == '__main__':
    main()



