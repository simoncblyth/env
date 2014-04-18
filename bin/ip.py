#!/usr/bin/env python
import socket

def address():
    """ 
    Not a general solution, but working for me 

    http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    """
    return socket.gethostbyname(socket.gethostname())


if __name__ == '__main__':
    print address()


