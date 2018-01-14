#!/usr/bin/env python
"""
Uses XMLRPC to Dumps all Trac wiki pages from potentially remote
trac instances into local files.

Usage, assuming config sections of `~/.env.cnf` named according to argument::

   tracwikidump.sh workflow_trac
   tracwikidump.sh env_trac 
   tracwikidump.sh heprez_trac 
   tracwikidump.sh tracdev_trac 

Actions of the script:

#. XMLRPC pulls the list of all wiki pages 
#. grabs pages individually and writes into local files in the configured 
   or current directory named "PageName.txt".  
#. Pre-existing files are not re-pulled, no version update checks are done
   (tracwiki is regarded as a dead source)


Config requirements
~~~~~~~~~~~~~~~~~~~

#. `~/.env.cnf` with credentialized xmlrpc login urls , eg::

    [workflow_trac]
    xmlrpc_url = http://USER:PASS@localhost/tracs/workflow/login/xmlrpc
    tracdump_outd = /usr/local/workflow/tracdump


Hmm whatabout tickets
~~~~~~~~~~~~~~~~~~~~~~~

To see API docs, visit url endpoint interactively, ignore phishing warning

After peering at the form of tickets data, conclusing that using 
the bitbucket export format would be good for a standard way to persist 
tickets.



Wikipage requirements
~~~~~~~~~~~~~~~~~~~~~~~

#. case degenerate page names not allowed, delete one of the degenerates


Dump Status
~~~~~~~~~~~~~

==========  =================================================================
 repo         nodes with dumps eg /usr/local/env/tracwikidump/
==========  =================================================================
 workflow     D G
 env          D G
 heprez       D
 tracdev      D
----------  -----------------------------------------------------------------
 aberdeen
 data
 newtest      
==========  =================================================================


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


DECIDED AGAINST
~~~~~~~~~~~~~~~

Tracwiki is now sufficiently dead, that simple .txt file dumps are enough.

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

def trac_wiki_dump( server , outd=".", dbgpages=[]):
   """
   :param server:
   :param dbgpages: list of pages on which to use a monkey patched ExpatParser with extra debug

   As no longer regard tracwiki as "live" no need to 
   bother with getting info and checking for any updates.
  
   So the dump becomes a once only operation, providing a 
   directory of .txt files for future grepping and "manual" conversion
   into Sphinx/docutils RST if deemed to be useful.
   """
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


def trac_ticket_dump( server, outd="."):
    """
    ::

        In [7]: server.ticket.get(60)
        Out[7]: 
        [60,
         <DateTime '20120903T12:11:03' at 110c5acb0>,
         <DateTime '20120903T12:12:59' at 110c5acf8>,
         {'_ts': '2012-09-03 12:12:59+00:00',
          'cc': '',
          'component': 'Admin',
          'description': "Hazily remembered issue, but make a ticket to capture some screen captures\nRecall that '''Mail.app''' fails to connect to hep1 (with no visible error) following \nsome intervention on hep1.\n\nSubsequently found solution was to check some dialog box always \naccepting hep1 certificates\n\n\n[[Image(mail-trust-hep1-dialog.png)]]\n\n\n[[Image(mail-certificate-trust-settings.png)]]",
          'keywords': 'Mail hep1 certificate',
          'milestone': '',
          'owner': 'blyth',
          'priority': 'major',
          'reporter': 'blyth',
          'resolution': 'fixed',
          'status': 'closed',
          'summary': 'recording ancient issue with Mail.app and hep1 certificates here',
          'type': 'defect',
          'version': ''}]

    """
    tkts = sorted(server.ticket.query("max=0"))  # id numbers of all tickets
    #import IPython
    #IPython.embed()

    for tk in tkts:
        atk = server.ticket.get(tk)
        assert len(atk) == 4
        id_, time_created, time_changed, attributes = atk
        assert id_ == tk
        assert time_created.__class__ == xmlrpclib.DateTime
        assert time_changed.__class__ == xmlrpclib.DateTime
        assert attributes.__class__ == dict
        assert len(attributes) == 14
        log.info("tkt %s %s %s %s " % (tk, time_created,time_changed, attributes['summary'] ))



def main():
   logging.basicConfig(level=logging.INFO)
   from env.web.cnf import cnf_ 
   cnf = cnf_(__doc__)   # argument parsed in cnf_ 
   url = cnf['xmlrpc_url']
   outd = cnf.get('tracdump_outd', cnf.outd)

   server = ServerProxy(url)
   log.info("API version %s " % server.system.getAPIVersion())

   trac_wiki_dump(server, os.path.join(outd,'wiki') , dbgpages=[])
   #trac_ticket_dump(server, os.path.join(outd,'ticket') )


if __name__ == '__main__':
    main()


