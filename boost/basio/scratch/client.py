#!/usr/bin/python         

import sys, socket, binascii 
x_ = lambda _:binascii.hexlify(_)

s = socket.socket()         
host = socket.gethostname() 
port = 50013               

s.connect((host, port))
r = s.recv(1024)
print(x_(r))

s.close()    




