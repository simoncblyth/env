#!/usr/bin/env python

import socket
import logging
log = logging.getLogger(__name__)
from glumpy.window import event


class DAEDispatcher(event.EventDispatcher):
    """
    http://www.pyglet.org/doc/programming_guide/creating_your_own_event_dispatcher.html
    """
    def __init__(self, port="15006", host="127.0.0.1"):
        """
        :param port:
        :param host: default of 127.0.0.1 restricts to local (same machine) connections only 
        """
        if host == "AUTO_HOST":
            host = socket.gethostname()
            log.warn("%s binding to potentially externally accessible ip:port %s:%s for UDP " % ( self.__class__.__name__, host, port )) 
        else:
            log.debug("%s binding to host:port %s:%s for UDP " % ( self.__class__.__name__, host, port ))

        self.port = port
        self.host = host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)
        self.sock.bind((host,int(port)))

    def __repr__(self):
        return "%s %s:%s " % ( self.__class__.__name__, self.host, self.port )

    def update(self):
        try:
            data, addr = self.sock.recvfrom(1024)
            self.received_message(data)
        except socket.error:
            pass   

    def received_message(self, msg ):
        log.debug("received_message %s " % msg )
        self.dispatch_event('on_external_message', msg)

    def on_external_message(self, msg):
        pass
        log.debug("default on_external_message %s " % msg )

DAEDispatcher.register_event_type('on_external_message')



