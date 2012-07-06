#!/usr/bin/env python
"""
I guess it should not matter which python os being used

Getting::

	xmlrpclib.Fault: <Fault 403: 'XML_RPC privileges are required to perform this operation'>

"""
import os
import xmlrpclib

if __name__ == '__main__':
   url = os.environ['TRAC_ENV_XMLRPC']	
   print url
   server = xmlrpclib.ServerProxy(url)
   pages = server.wiki.getAllPages()
   for page in pages:
       print page	   


