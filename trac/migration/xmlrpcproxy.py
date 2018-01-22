#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
from xmlrpclib import ServerProxy
from env.web.cnf import Cnf


class Proxy(object):

    @classmethod
    def create(cls, sect="workflow_trac", cnfpath="~/.env.cnf"):
        cnf = Cnf.read(sect, cnfpath)
        return cls(cnf)

    def __init__(self, cnf):
        proxy = ServerProxy(cnf['xmlrpc_url'])
        log.info("API version %s " % proxy.system.getAPIVersion())
        pages = proxy.wiki.getAllPages()  
        log.info("found %s pages " % len(pages))

        self.cnf = cnf 
        self.proxy = proxy
        self.pages = sorted(pages)

    def __repr__(self):
        return "<Proxy %s pages %d> " % (self.cnf, len(self.pages))

 
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)
    prx = Proxy.create()
    print prx    

    print "\n".join(prx.pages)


