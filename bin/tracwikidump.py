#!/usr/bin/env python
"""
Usage::

   tracwikidump.py workflow_trac

Uses XMLRPC to communicate with a potentially remote Trac instance identified
via a URL configured in a named section of the config :file:`~/.env.cnf` 
under key `xmlrpc_url` with values of form::

   http://USER:PASS@localhost/tracs/workflow/login/xmlrpc

The script proceeds to:

#. pulls the list of all wiki pages 
#. grabs them individually and writes into local files in the current directory
   named after the wikipage with a ".txt" extension. 
   Pre-existing files are not re-pulled

Trac requirements
~~~~~~~~~~~~~~~~~~

This requires two Trac plugins to be installed and configured:

#. TracXMLRPC
#. TracHttpAuth

See `tracxmlrpc-` and `trachttpauth-` for installation of those.


TRAC_ENV_XMLRPC envvar
~~~~~~~~~~~~~~~~~~~~~~~

Formerly the :envvar:`TRAC_ENV_XMLRPC` was used to define the target instance.
This can be defined with::

   trac-  
   tracxmlrpc-  # uses the default TRAC_INSTANCE name for the node

Issues
~~~~~~~

#. Trac allows multiple page name differing only in case, but filesystem doesnt

   #. throwing exceptions currently, solution is to rename to eliminate degeneracy

#. Trac allows some non-ascii chars within wiki pages, that xmlrpclibs expat parser barfs with::

   xml.parsers.expat.ExpatError: not well-formed (invalid token): line 356, column 12
   
#. ascii escapes hex 1b ( represented in vi with ``^[`` ) also cause expat parser barfs::

    g4pb-2:twd blyth$ hexdump char.txt 
    0000000 61 61 62 62 63 63 1b 0a 61 61 62 62 63 63 1b 0a


"""
import os, logging
import xmlrpclib
from xmlrpclib import ServerProxy, ExpatParser
log = logging.getLogger(__name__)

class DbgExpatParser(ExpatParser):
    """
    Monkey patch for parser used by xmlrpclib
    """
    def feed(self, data):
        print "DbgExpatParser:\n\n%s\n\n" % data     
        self._parser.Parse(data, 0)

def check_case_degeneracy( pages ):
    """
    Check for case degenerate page names
    """
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
        msg = "case degerate trac wiki page names detected %r " % degen 
        log.fatal(msg)
        raise Exception(msg)  

def tracwikidump( url , dbgpages=[]):
   """
   :param url:
   :param dbgpages: list of pages on which to use a monkey patched ExpatParser with extra debug
   """
   log.debug("tracwikidump %s " % url)
   server = ServerProxy(url)
   log.info("API version %s " % server.system.getAPIVersion())
   pages = server.wiki.getAllPages()
   npage = len(pages) 
   log.info("pages %s %s " % ( npage, pages )) 
   check_case_degeneracy(pages)

   for i, page in enumerate(sorted(pages)):
       if page in dbgpages:
           log.warn("extra debug for  %s " % page )
           xmlrpclib.ExpatParser = DbgExpatParser
       else:
           xmlrpclib.ExpatParser = ExpatParser

       path = "%s.txt" % page
       if os.path.exists(path):
           log.info("skip preexisting file %s " % path )    
       else:
           log.info("getting page %s " % path )
           content=server.wiki.getPage(page)
           out=file(path,'w')
           out.write( content.encode("utf-8"))
           out.close()


if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO)
   from env.web.cnf import cnf_ 
   cnf = cnf_(__doc__)
   url = cnf['xmlrpc_url']
   tracwikidump(url, dbgpages=[])



