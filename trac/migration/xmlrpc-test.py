#!/usr/bin/env python
"""
I guess it should not matter which python os being used

Getting 404::

   ./xmlrpc-test.py http://dayabay.phys.ntu.edu.tw/tracs/env/rpc/
   ./xmlrpc-test.py http://dayabay.phys.ntu.edu.tw/tracs/env/xmlrpc/

401 Authentication Required::

   ./xmlrpc-test.py http://localhost/tracs/workflow/xmlrpc/


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
   for page in sorted(pages):
       print page	   
       if os.path.exists(page):
           log.info("skip preexisting file %s " % page )    
       else:
           content=server.wiki.getPage(page)
           out=file(page,'w')
           out.write( content.encode("utf-8"))
           out.close()



