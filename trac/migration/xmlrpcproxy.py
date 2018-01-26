#!/usr/bin/env python

import logging, sys
log = logging.getLogger(__name__)
import xmlrpclib 
import socket
from env.web.cnf import Cnf

class Proxy(object):
    @classmethod
    def create(cls, sect="workflow_trac", cnfpath="~/.workflow.cnf"):
        cnf = Cnf.read(sect, cnfpath)
        return cls(cnf)

    def make_proxy(self, cnf):
        proxy = xmlrpclib.ServerProxy(cnf['xmlrpc_url'])
        try:
            proxy.fictional_method()
        except xmlrpclib.Fault:
            pass
        except socket.error:
            proxy = None
        pass
        return proxy

    def __init__(self, cnf):
        self.cnf = cnf 
        proxy = self.make_proxy(cnf)
        if proxy is None:
            log.warning("failed to create xmlrpclib proxy, perhaps server not running")
            pages = []
        else:  
            log.info("API version %s " % proxy.system.getAPIVersion())
            pages = proxy.wiki.getAllPages()  
            log.info("found %s pages " % len(pages))
        pass

        if proxy is None:
            log.fatal("ABORT : failed to create proxy to Trac server") 
            sys.exit(1)
        pass

        self.proxy = proxy
        self.pages = sorted(pages)

    def __repr__(self):
        return "<Proxy %s pages %d> " % (self.cnf, len(self.pages))

 
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    prx = Proxy.create()
    print prx    

    print "\n".join(prx.pages)


