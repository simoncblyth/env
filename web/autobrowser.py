#!/usr/bin/env python
"""

"""
from __future__ import with_statement
import os, logging, mechanize
from lxml import etree
from lxml.etree import tostring
from StringIO import StringIO
from urlparse import urlparse

log = logging.getLogger(__name__)

def parse_( content ):
    return etree.parse( StringIO(content), etree.HTMLParser() ).getroot()

def serialize_(tree, path):
    with open(path,"w") as fp:fp.write(tostring(tree))


class AutoBrowser(object):
    """
    Automated and authenticated browser, with html parsing 
    """
    def __init__(self, cnf, useragent='Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1'):
        """
        :param cnf:
        :param useragent:

        Instanciation creates mechanize Browser and authenticates
        using form and/or basic methods. 

        #. circumvent robots.txt restriction by pretending to be a browser
        """
        br  = mechanize.Browser()
        br.set_handle_robots(False)
        br.addheaders = [('User-agent', useragent )]
        #br.addheaders = [('X_REQUESTED_WITH','XMLHttpRequest')]

        self._basic( br, cnf )
        self._form( br, cnf )
        self.cnf = cnf 
        self.br = br

    def _basic(self, br, cnf):
        """
        BASIC http authentication

        :param br: mechanize browser instance
        :param cnf: Cnf instance

        """
        if cnf.get('basic_url',None):
            log.debug('basic hookup')
            br.add_password(cnf['basic_url'], cnf['basic_user'], cnf['basic_pass'] )
        else:    
            log.debug("basic_url not configured, skip basic hookup")

    def _form(self, br, cnf):
        """
        FORM http authentication

        Note that sometimes forms are present despite not being displayed in 
        ordinary browsers (eg search panels made invisible on some pages).

        The advantage in using mechanize over lower level approaches is that it 
        handles the form tokens. 

        :param br: mechanize browser instance
        :param cnf: Cnf instance

        """
        form_url = cnf.get('form_url',None)
        if not form_url:return
        log.debug("form authentication to %s " % form_url )
        br.open(form_url)
        br.select_form(nr=int(cnf['form_nr']))
        f = br.form
        f[cnf['form_userkey']] = cnf['form_user']
        f[cnf['form_passkey']] = cnf['form_pass']
        br.submit()
        html = br.response().read()

    def open_(self, url, parse=False):
        """
        :param url:
        :param parse:
        :return: parsed tree of the html content returned from the url, or None if parse is False

        .. warn: changed behavior, with parse False formerly returned None now return raw content 

        """
        log.debug("open_ %s " % url )
        self.br.open(url)
        html = self.br.response().read()
        if parse:
            tree = parse_( html )
            return tree
        return html    

    def outd_(self, target):
        """
        :param target: URL of page with the links to harvest from
        :return: output directory based on configured base and url, or overridden by commandline option
        """
        if self.cnf.outd:return self.cnf.outd
        tmpd = self.cnf['tmpd']
        urlt = urlparse(target)
        host = urlt.hostname
        path = urlt.path
        assert path.startswith('/') and len(path)>10, "invalid path extracted from url %s %s " % ( path, urlt )
        outd = "%s/%s%s" % (tmpd,host, path) 
        return outd

    def chdir_(self, target):
        """
        Create output directory corresponding to a target URL and change directory to it

        :param target: URL 
        :return: output directory 
        """
        outd = self.outd_(target)
        if not os.path.isdir(outd):
            log.info("creating output dir %s " % outd )
            os.makedirs(outd)
        else:
            log.info("retreiving into pre-existing dir %s " % outd ) 
        pass    
        os.chdir(outd)    
        return outd


    def retrieve(self, url, filename ):
        """
        Download from url, saving into the filename 

        :param url:
        :param filename:
        """ 
        try:
            ret = self.br.retrieve( url, filename=filename )
        except mechanize.HTTPError:
            log.debug("retrieve error for %s %s " % ( filename, url))
            ret = None
        return ret    

if __name__ == '__main__':
    pass 
    from cnf import cnf_
    cnf = cnf_(__doc__)
    ab = AutoBrowser(cnf)


