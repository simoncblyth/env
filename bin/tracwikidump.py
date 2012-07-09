#!/usr/bin/env python
"""
Uses XMLRPC to communicate with a potentially remote Trac instance.
Pulling all wiki pages and writing into local files in the current directory
named after the wikipage with a ".txt" extension

The Trac instance is identified via a URL in envvar TRAC_ENV_XMLRPC such as::

   http://USER:PASS@localhost/tracs/workflow/login/xmlrpc

This can be defined with::

   trac-	  
   tracxmlrpc-  # uses the default TRAC_INSTANCE name for the node

This requires a two plugins to be installed and configured:

#. TracXMLRPC
#. TracHttpAuth

See *tracxmlrpc-* and *trachttpauth-* for installtion of those.


Issues::

#. Trac allows two page name differing only in case, but filesystem doesnt

   #. throwing exceptions currently, solution is to rename to eliminate degeneracy

#. Trac allows some non-ascii chars within wiki pages, that xmlrpclibs expat parser barfs with::

   xml.parsers.expat.ExpatError: not well-formed (invalid token): line 356, column 12
   
#. ascii escapes hex 1b ( represented in vi with ``^[`` ) also cause expat parser barfs::

	g4pb-2:twd blyth$ hexdump char.txt 
	0000000 61 61 62 62 63 63 1b 0a 61 61 62 62 63 63 1b 0a
	*
	0000010
	g4pb-2:twd blyth$ cat char.txt 
	aabbcc
	      aabbcc
			  aabbcc



"""
import os, logging
import xmlrpclib
from xmlrpclib import ServerProxy, ExpatParser
log = logging.getLogger(__name__)


class DbgExpatParser(ExpatParser):
    def feed(self, data):
	print "DbgExpatParser:\n\n%s\n\n" % data     
	self._parser.Parse(data, 0)


def check_case_degeneracy( pages ):
    lpages = map( str.lower, pages )
    degen = []
    for page in pages:
        lpage = str.lower(page)
        dpage = filter( lambda _:str.lower(_) == lpage , lpages )
        if len(dpage) != 1:
	    log.warn("degenerate %s %s " % ( page, repr(dpage) ))     
	    degen.append(page)
	pass    
    if len(degen)>0:
        msg = "case degerate trac wiki page names detected " 
        log.fatal(msg)
        raise Exception(msg)  



def tracwikidump( url , skips=[]):
   print url
   server = ServerProxy(url)
   print server.system.getAPIVersion()
   pages = server.wiki.getAllPages()
   check_case_degeneracy(pages)

   for page in sorted(pages):
       log.debug(page)
       if page in skips:
	   log.warn("skipping %s " % page )
           xmlrpclib.ExpatParser = DbgExpatParser
       else:
           xmlrpclib.ExpatParser = ExpatParser


       path = "%s.txt" % page
       if os.path.exists(path):
           log.debug("skip preexisting file %s " % path )    
       else:
           content=server.wiki.getPage(page)
           out=file(path,'w')
           out.write( content.encode("utf-8"))
           out.close()


if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO)	
   import sys	
   if len(sys.argv)>1:
       url = sys.argv[1]
   else:    
       url = os.environ['TRAC_ENV_XMLRPC']	


   tracwikidump(url, skips=[])




