#!/usr/bin/env python
"""
Status
-------

#. succeeds to mechanically login to dybsvn Trac, for automated response time measurements


"""
from __future__ import with_statement
import mechanize, logging, os, re
from ConfigParser import ConfigParser 
from pprint import pformat
from urlparse import urlparse
from lxml import etree
from lxml.etree import tostring
from StringIO import StringIO

from shiftcheck import Visitor


log = logging.getLogger(__name__)

def cnf_(path, siteconf):
    cpr=ConfigParser()
    cpr.read(os.path.expanduser(path))
    return dict(cpr.items(siteconf))

def parse_( content ):
    return etree.parse( StringIO(content), etree.HTMLParser() ).getroot()


class Browser(object):
    """
    Expanding mechanize access to new sites typically requires
    ipython interactive sessions to determine how to gain automatic access.


    https://views.scraperwiki.com/run/python_mechanize_cheat_sheet/?
    """
    def __init__(self, cnf ):
        br  = mechanize.Browser()

        # circumvent robots.txt restriction by pretending to be a browser
        br.set_handle_robots(False)
        br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
        #br.addheaders = [('X_REQUESTED_WITH','XMLHttpRequest')]

        self._basic( br, cnf )
        self._form( br, cnf )
        self.cnf = cnf 
        self.br = br
        self.links = []
        self.tree = None

        if cnf.get('visitor',None):
            path = os.path.expandvars(cnf['visitor'])
            visitor = Visitor(path=path,aprefix=cnf['visitor_aprefix'])
        else:     
            visitor = None
        self.visitor = visitor    

    def _basic(self, br, cnf):
        if cnf['basic_url']:
            log.info('basic hookup')
            br.add_password(cnf['basic_url'], cnf['basic_user'], cnf['basic_pass'] )
        else:    
            log.warn("basic_url not configured, skip basic hookup")

    def _form(self, br, cnf):
        """
        Note that sometimes forms are present despite not being displayed in 
        ordinary browsers::

            In [24]: for i,f in enumerate(br.forms()):print i,f,f.method
               ....: 
            0 <GET http://dayabay.ihep.ac.cn/tracs/dybsvn/search application/x-www-form-urlencoded> GET
            1 <POST http://dayabay.ihep.ac.cn/tracs/dybsvn/login/ application/x-www-form-urlencoded
              <HiddenControl(__FORM_TOKEN=6f763fd06c6da16665cbef6c) (readonly)>
              <HiddenControl(referer=) (readonly)>
              <TextControl(user=)>
              <PasswordControl(password=)>
              <SubmitControl(<None>=Login) (readonly)>> POST

        Note that the mechanize advantage of holding the form tokens 
        """
        if not cnf.get('form_url',None):return
        br.open(cnf['form_url'])
        br.select_form(nr=int(cnf['form_nr']))
        f = br.form
        #print f
        f[cnf['form_userkey']] = cnf['form_user']
        f[cnf['form_passkey']] = cnf['form_pass']
        br.submit()
        html = br.response().read()
        #print html


    def shiftcheck(self):
        assert self.visitor
        takes = []
        skips = dict(take_ptn=[],skip_ptn=[],preexist=[],ext=[],start=[],httperror=[])
        for n,item in enumerate(self.visitor):
            #if n>1:continue
            print "%-4s %-60s %s " % ( n, item.name, item['href'][0:100])
            name = "%s.png" % item.name
            url = item['href']
            try:
                log.info("retrieve %s %s " % ( name, url ))

                tree = self.open_(url, parse=True)
                imgs = tree.xpath('.//img')
                assert len(imgs) == 3, imgs
                #for img in imgs:
                #    print tostring(img)
                src = imgs[-1].attrib['src']
                usrc = "http://dcs2.dyb.ihep.ac.cn/%s" % src
                print usrc

                r = self.br.retrieve( usrc, filename=name )
                takes.append(name)
            except mechanize.HTTPError:
                log.debug("retrieve error for %s %s " % ( name, url))
                skips['httperror'].append(name)
        self.skips = skips
        self.takes = takes
        log.info("SKIPS\n%s\n" % pformat(skips))
        log.info("TAKES\n%s\n" % pformat(takes))

 
    def links_(self, skip_ptn="^NOTHING" , take_ptn=".*", exts=[] , start=None ):
        """

        In [4]: l.absolute_url
        Out[4]: 'http://neutrino2.physics.sjtu.edu.cn/DQTests/figures_last_8_hours/DayaBayAD1_Flasher_2inchPMT.png'

        In [6]: l.base_url
        Out[6]: 'http://neutrino2.physics.sjtu.edu.cn/DQTests/figures_last_8_hours/webpage.EH1.html'

        In [7]: l.tag
        Out[7]: 'a'

        In [8]: l.text
        Out[8]: 'AD1 2inchPMT Flashing Rate'

        In [9]: l.url
        Out[9]: 'DayaBayAD1_Flasher_2inchPMT.png'

        """ 


        takes = []
        skips = dict(take_ptn=[],skip_ptn=[],preexist=[],ext=[],start=[],httperror=[])
        skip = re.compile(skip_ptn) if skip_ptn else None 
        take = re.compile(take_ptn) if take_ptn else None

        links = self.br.links()
        for link in links:
            label = link.text
            url = link.absolute_url
            name = os.path.basename(url)
            base,ext = os.path.splitext(name)

            if os.path.exists(name):
                log.debug("pre-existing %s %s " % ( name, url ))
                skips['preexist'].append(name)
                continue
            if len(exts) > 0 and ext not in exts:
                log.debug("wrong extension %s %s %s " % ( ext, exts, name ))
                skips['ext'].append(name)
                continue
            if skip and skip.match(name):
                log.debug("skip_ptn %s skipping url %s " % ( skip_ptn, name ))
                skips['skip_ptn'].append(name)
                continue
            if take and not take.match(name):
                log.debug("take_ptn %s not-taking url %s " % ( take_ptn, name ))
                skips['take_ptn'].append(name)
                continue
            if start and not url.startswith(start):
                log.debug("start %s not-taking url %s " % ( start, url ))
                skips['start'].append(name)
                continue
            try:
                log.info("retrieve %s %s " % ( name, url ))
                #r = self.br.retrieve( url, filename=name )
                takes.append(name)
                self.links.append(link)
            except mechanize.HTTPError:
                log.debug("retrieve error for %s %s " % ( name, url))
                skips['httperror'].append(name)
        self.skips = skips
        self.takes = takes
        log.info("SKIPS\n%s\n" % pformat(skips))
        log.info("TAKES\n%s\n" % pformat(takes))

    def open_(self, url, parse=False):
        log.info("opening %s " % url )
        self.br.open(url)
        if parse:
            html = self.br.response().read()
            tree = parse_( html )
        else:
            tree = None
        self.tree = tree
        return tree


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #site = 'dybsvn_trac'
    site = 'shiftcheck'
    #site = 'dqtests'
    os.environ.setdefault('SITECONF',site)
    siteconf = os.environ['SITECONF']
    cnf =  cnf_("~/.env.cnf", siteconf )
    print pformat(cnf)
    br = Browser( cnf )

    targets = cnf.get('targets',"").split()
    for target in targets:
        urlt = urlparse(target)
        host = urlt.hostname
        path = urlt.path
        assert path.startswith('/') and len(path)>10, "invalid path extracted from url %s %s " % ( path, urlt )
        outd = "/tmp/env/web/%s%s" % (host, path) 
        if not os.path.isdir(outd):
            log.info("creating output dir %s " % outd )
            os.makedirs(outd)
        else:
            log.info("retreiving into pre-existing dir %s " % outd ) 
        pass    
        os.chdir(outd)    
        br.open_(target, parse=True)

        if siteconf == 'shiftcheck':
            br.shiftcheck()
        else:    
            br.links_(skip_ptn=cnf.get('skip_ptn',None), take_ptn=cnf.get('take_ptn',None), exts=cnf.get('exts',"").split() )






