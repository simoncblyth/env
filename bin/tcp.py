#!/usr/bin/env python
import os, sys, socket
"""


"""

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = os.environ.get("TCP_PORT","15006")
    host = socket.gethostname()
    #host = os.environ.get("TCP_HOST","127.0.0.1")

    print("sock.connect to host:port %s:%s " % (host, port))
    sock.connect((host, int(port)))
    
    arg = " ".join(sys.argv[1:])
    print("sock.sendall [%s] " % (arg))
    sock.sendall(arg.encode())
    data = sock.recv(4096)
    sock.close()

    print("received %s " % data)



if __name__ == '__main__':
    main()



