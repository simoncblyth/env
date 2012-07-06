#!/usr/bin/env python
"""
I guess it should not matter which python os being used

Getting::

	xmlrpclib.Fault: <Fault 403: 'XML_RPC privileges are required to perform this operation'>

"""
import sys, os
from xmlrpclib import ServerProxy 

if __name__ == '__main__':

   if len(sys.argv)>1:
       url = sys.argv[1]
   else:    
       url = os.environ['TRAC_ENV_XMLRPC']	

   print url
   server = ServerProxy(url)
   print server.system.getAPIVersion()
   pages = server.wiki.getAllPages()
   for page in pages:
       print page	   


