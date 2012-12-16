#!/usr/bin/env python
"""

Mechanized Browsing
=======================

::

   ./mech.py shiftcheck -n 6          # for the hourly check       
   ./mech.py shiftcheck -n 1000       # for the 4-hour check

   ./mech.py shiftcheck -n 6 -o /tmp/1hr    # convenient to write 1hr PNGs in separate folder

   open /tmp/1hr          # coverflow in Finder is sufficient for 6 PNGs

   open file:///tmp/env/web/dayawane.ihep.ac.cn/twiki/bin/view/Internal/ShiftCheck/annotated.html    

        # make sure to refer to the annotated page corresponding to the shiftcheck run to avoid confusion
        # arising from twiki updates 

   ./mech.py dqtests -n 10            
   ./mech.py dybsvn_trac              # needs shakedown


1hr Check
----------

::

    simon:web blyth$ ./mech.py shiftcheck -n 6 -o /tmp/1hr
    INFO:__main__:retreiving into pre-existing dir /tmp/1hr 
    INFO:shiftcheck:. 001_EH1__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_T1&ParaNames%5B%5D=DBNS_PTH_T2 ...
    INFO:shiftcheck:. 002_EH1__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_H1&ParaNames%5B%5D=DBNS_PTH_H2 ...
    INFO:shiftcheck:. 003_EH2__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_T1&ParaNames%5B%5D=LANS_PTH_T2 ...
    INFO:shiftcheck:. 004_EH2__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_H1&ParaNames%5B%5D=LANS_PTH_H2 ...
    INFO:shiftcheck:. 005_EH3__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FARS_PTH_T1&ParaNames%5B%5D=FARS_PTH_T2 ...
    INFO:shiftcheck:. 006_EH3__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FARS_PTH_H1&ParaNames%5B%5D=FARS_PTH_H2 ...
    INFO:__main__:STAT
    {1: '.', 2: '.', 3: '.', 4: '.', 5: '.', 6: '.'}

    INFO:__main__:wrote annotated target page to /private/tmp/1hr/annotated.html 
    simon:web blyth$ open /private/tmp/1hr/annotated.html                               # ShiftCheck with links indexed corresponding to above PNG name


4hr Check
-----------

::

    simon:web blyth$ time ./mech.py shiftcheck -n 1000 -o /tmp/4hr
    INFO:__main__:creating output dir /tmp/4hr 
    INFO:shiftcheck:. 001_EH1__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_T1&ParaNames%5B%5D=DBNS_PTH_T2 ...
    INFO:shiftcheck:. 002_EH1__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_H1&ParaNames%5B%5D=DBNS_PTH_H2 ...
    INFO:shiftcheck:. 003_EH2__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_T1&ParaNames%5B%5D=LANS_PTH_T2 ...
    INFO:shiftcheck:. 004_EH2__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_H1&ParaNames%5B%5D=LANS_PTH_H2 ...
    ...
    INFO:shiftcheck:. 238_EH3__Weights.png                                          http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=weight_isobutane&ParaNames%5B%5D=weight ...
    INFO:shiftcheck:. 239_EH3__Pressures.png                                        http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=pressure_argon&ParaNames%5B%5D=pressure ...
    INFO:shiftcheck:. 240_EH1_EH1_RPC_VME_Temperature.png                           http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&ParaNames%5B%5D=Temperat ...
    INFO:shiftcheck:. 241_EH2_EH2_RPC_VME_Temperature.png                           http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&ParaNames%5B%5D=Temperat ...
    INFO:shiftcheck:. 242_EH3_EH3_RPC_VME_Temperature.png                           http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&TimeSpan=1440&Interval=1 ...
    INFO:shiftcheck:. 243_SAB_Temperature___Last_30_Minutes.png                     http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_SAB_Temp_PT1&ParaNames%5B%5D=DBNS_ ...
    INFO:shiftcheck:. 244_VME_Crate_Temperature___Last_30_Minutes.png               http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&ParaNames%5B%5D=Temperat ...
    INFO:__main__:wrote annotated target page to /private/tmp/4hr/annotated.html 

    real    2m1.255s
    user    0m5.214s
    sys     0m1.989s


Status
-------

#. succeeds to mechanically login to dybsvn Trac, for automated response time measurements

Ideas
------

#. metadata-ize the retreived PNG to indicate where they came from ? 
   Instead took easier route of planting metadata and link indices in the target html 

#. split Browser from 



"""
from __future__ import with_statement
import mechanize, logging, os, re
from ConfigParser import ConfigParser 
from optparse import OptionParser
from pprint import pformat
from urlparse import urlparse
from lxml import etree
from lxml.etree import tostring
from StringIO import StringIO

from shiftcheck import Visitor


log = logging.getLogger(__name__)

def parse_( content ):
    return etree.parse( StringIO(content), etree.HTMLParser() ).getroot()

def serialize_(tree, path):
    with open(path,"w") as fp:fp.write(tostring(tree))

class Browser(object):
    """
    Expanding mechanize access to new sites typically requires
    ipython interactive sessions to determine how to gain automatic access.

    https://views.scraperwiki.com/run/python_mechanize_cheat_sheet/?
    """
    def __init__(self, cnf ):
        """
        #. circumvent robots.txt restriction by pretending to be a browser
        """
        br  = mechanize.Browser()
        br.set_handle_robots(False)
        br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
        #br.addheaders = [('X_REQUESTED_WITH','XMLHttpRequest')]

        self._basic( br, cnf )
        self._form( br, cnf )
        self.cnf = cnf 
        self.br = br

    def _basic(self, br, cnf):
        """
        BASIC http authentication
        """
        if cnf['basic_url']:
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
        """
        if not cnf.get('form_url',None):return
        br.open(cnf['form_url'])
        br.select_form(nr=int(cnf['form_nr']))
        f = br.form
        f[cnf['form_userkey']] = cnf['form_user']
        f[cnf['form_passkey']] = cnf['form_pass']
        br.submit()
        html = br.response().read()

    def shiftcheck(self, tree, pull=range(1) ):
        """
        :param tree: lxml parsed root node of html page
        :param pull: list of indices to pull
        """
        v = Visitor(tree,aprefix=self.cnf['visitor_aprefix'],pull=pull )
        v.retrieve_( self )
        log.info("STAT\n%s\n" % pformat(v))
        path = os.path.abspath("annotated.html")
        v.write_tree(path)
        log.info("wrote annotated target page to %s " % path )

    def retrieve(self, url, filename ):
        try:
            ret = self.br.retrieve( url, filename=filename )
        except mechanize.HTTPError:
            log.debug("retrieve error for %s %s " % ( filename, url))
            ret = None
        return ret    

    def links_(self, skip_ptn="^NOTHING" , take_ptn=".*", exts=[] , start=None ):
        """
        TODO: simplify/split off
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
                r = self.br.retrieve( url, filename=name )
                takes.append(name)
            except mechanize.HTTPError:
                log.debug("retrieve error for %s %s " % ( name, url))
                skips['httperror'].append(name)
        self.skips = skips
        self.takes = takes
        log.info("SKIPS\n%s\n" % pformat(skips))
        log.info("TAKES\n%s\n" % pformat(takes))

    def open_(self, url, parse=False):
        """
        :param url:
        :param parse:
        :return: parsed tree of the html content returned from the url, or None if parse is False
        """
        log.debug("opening %s " % url )
        self.br.open(url)
        if not parse:return None
        html = self.br.response().read()
        tree = parse_( html )
        return tree

    def outd_(self, target):
        """
        :param target: URL of page with the links to harvest from
        :return: output directory based on configured base and url, or overridden by commandline option
        """
        if cnf.outd:return cnf.outd
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


class Cnf(dict):pass


def cnf_(doc):
    parser = OptionParser(doc)
    parser.add_option("-c", "--cnfpath", help="file from which to load config setting.", default="~/.env.cnf" )
    parser.add_option("-l", "--level", help="loglevel", default="INFO" )
    parser.add_option("-n", "--npull", type=int, help="restrict retreival to first n links", default=10 )
    parser.add_option("-o", "--outd", default=None, help="directory in which to output retreived files, the default is based on configured base and the target url")
    (opts, args) = parser.parse_args()
    logging.basicConfig(level=getattr(logging,opts.level.upper()))
    assert len(args) == 1, "must supply siteconf section name present in %s " % opts.cnfpath
    cpr=ConfigParser()
    site = args[0]
    cpr.read(os.path.expanduser(opts.cnfpath))
    d = Cnf(cpr.items(site))
    d.site = site
    d.outd = opts.outd
    d.npull = opts.npull
    return d

if __name__ == '__main__':
    cnf = cnf_(__doc__)
    br = Browser( cnf )
    targets = cnf.get('targets',"").split()

    for target in targets:
        outd = br.chdir_(target)
        tree = br.open_(target, parse=True)
        if cnf.site == 'shiftcheck':
            br.shiftcheck(tree, pull=range(1,cnf.npull+1) )   # link indices count from 1, for better human interface
        else:    
            br.links_(skip_ptn=cnf.get('skip_ptn',None), take_ptn=cnf.get('take_ptn',None), exts=cnf.get('exts',"").split() )




