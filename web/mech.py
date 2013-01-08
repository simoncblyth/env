#!/usr/bin/env python
"""

"""
import logging, os, re
from pprint import pformat
from autobrowser import AutoBrowser
from shiftcheck import Visitor
log = logging.getLogger(__name__)

class Retriever(AutoBrowser):

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

if __name__ == '__main__':

    from cnf import cnf_
    cnf = cnf_(__doc__)
    br = Retriever( cnf )
    targets = cnf.get('targets',"").split()
    for target in targets:
        outd = br.chdir_(target)
        tree = br.open_(target, parse=True)
        br.links_(skip_ptn=cnf.get('skip_ptn',None), take_ptn=cnf.get('take_ptn',None), exts=cnf.get('exts',"").split() )


