#!/usr/bin/env python
"""
Usage::

   tracwikidump.py workflow_trac
   tracwikidump.py env_trac 

Uses XMLRPC to communicate with a potentially remote Trac instance identified
and pulls all wiki pages into local files.

Actions of the script:

#. pull the list of all wiki pages 
#. grabs pages individually and writes into local files in the configured 
   or current directory named "PageName.txt".  
   Pre-existing files are not re-pulled

Config requirements
~~~~~~~~~~~~~~~~~~~

#. `~/.env.cnf` with credentialized xmlrpc login urls , eg::

    xmlrpc_url = http://USER:PASS@localhost/tracs/workflow/login/xmlrpc
    tracwikidump_outd = /usr/local/workflow/tracwikidump

Wikipage requirements
~~~~~~~~~~~~~~~~~~~~~~~

#. case degenerate pages not allowed, delete one of them 



Dump Status
~~~~~~~~~~~~~

==========  =================================================================
 repo         dump status
==========  =================================================================
 workflow     D:/usr/local/workflow/tracwikidump/
 env          D:/usr/local/env/tracwikidump/ 
 heprez       D:/usr/local/heprez/tracwikidump/
 tracdev      D:/uar/local/tracdev/tracwikidump/
----------  -----------------------------------------------------------------
 aberdeen
 data
 newtest      
==========  =================================================================

Issues
~~~~~~~~

* http://dayabay.phys.ntu.edu.tw/tracs/env/login/xmlrpc

No handler matched request to /login/xmlrpc


Trac requirements
~~~~~~~~~~~~~~~~~~

This requires two Trac plugins to be installed and configured:

#. TracXMLRPC
 
   * without this installed and enabled get protocol 404 errors 
   * without permissions setup get::

     Fault 403: 'XML_RPC privileges are required to perform this operation'

#. http://trac-hacks.org/wiki/HttpAuthPlugin

   * workaround allowing accountmanager/xmlrpc to interop

See `tracxmlrpc-` and `trachttpauth-` for installation of those.



TRAC_ENV_XMLRPC envvar
~~~~~~~~~~~~~~~~~~~~~~~

Formerly the :envvar:`TRAC_ENV_XMLRPC` was used to define the target instance.
This can be defined with::

   trac-  
   tracxmlrpc-  # uses the default TRAC_INSTANCE name for the node


TODO
~~~~~

#. propagate metadata about wiki pages, maybe into a sidecar JSON file

   #. tags
   #. last modification time
   #. last author 
   #. drop the wikipage history of changes : not sufficiently useful to preserve

       * http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/WikiStart?action=history

#. does the XMLRPC API provide access to this information ?

   #. no access to tags 
   #. getPageInfo provides a dict::

    {'comment': '', 'lastModified': <DateTime '20100405T06:11:50' at 70ccb0>, 'version': 1, 'name': 'ArcDjimg', 'author': 'blyth'}

   #. maybe do twostep of smuggling tags in the comment field 
   #. OR directly via SQL query interface to trac

       * http://localhost/tracs/workflow/report/12?format=csv&NAME=PageName&USER=blyth  works in browser, kicked to login from curl


info via SQL reports
~~~~~~~~~~~~~~~~~~~~~~~

* http://localhost/tracs/workflow/report/12?format=csv&USER=blyth

   * need to defeat pagination, try 

::

   -- @LIMIT_OFFSET@

* if cannot do so would need to grab tags for each page 






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
        import IPython
        IPython.embed()
        #raise Exception(msg)  

def tracwikidump( url , outd=".", dbgpages=[]):
   """
   :param url:
   :param dbgpages: list of pages on which to use a monkey patched ExpatParser with extra debug

   As no longer regard tracwiki as "live" no need to 
   bother with getting info and checking for any updates.
  
   So the dump becomes a once only operation, providing a 
   directory of .txt files for future grepping and "manual" conversion
   into Sphinx/docutils RST if deemed to be useful.
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

       path = os.path.join(outd,"%s.txt" % page)
       odir = os.path.dirname(path)

       if not os.path.exists(odir):
           log.info("creating directory %s " % odir )
           os.makedirs(odir)

       if os.path.exists(path):
           log.info("skip preexisting file %s " % path )    
       else:
           info=server.wiki.getPageInfo(page)
           log.info("getting page %s info %s " % (path,repr(info)) )
           content=server.wiki.getPage(page)
           out=file(path,'w')
           out.write( content.encode("utf-8"))
           out.close()


def main():
   logging.basicConfig(level=logging.INFO)
   from env.web.cnf import cnf_ 
   cnf = cnf_(__doc__)
   url = cnf['xmlrpc_url']
   outd = cnf.get('tracwikidump_outd', cnf.outd)
   tracwikidump(url, outd , dbgpages=[])


if __name__ == '__main__':
    main()


