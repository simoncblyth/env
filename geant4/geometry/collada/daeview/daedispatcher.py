#!/usr/bin/env python

import socket
from glumpy.window import event

class DAEDispatcher(event.EventDispatcher):
    """
    http://www.pyglet.org/doc/programming_guide/creating_your_own_event_dispatcher.html
    """
    def __init__(self, port=15006, ip="127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.sock.bind((ip,port))

    def update(self):
        try:
            data, addr = self.sock.recvfrom(1024)
            self.received_message(data)
        except socket.error:
            pass   

    def received_message(self, msg ):
        #print "received_message %s " % msg 
        self.dispatch_event('on_external_message', msg)

    def on_external_message(self, msg):
        pass
        #print "default on_external_message %s " % msg 

DAEDispatcher.register_event_type('on_external_message')



