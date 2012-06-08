#!/usr/bin/env python
"""
Match precursor lines like::

   safari-(){      . $(workflow-home)/tools/safari.bash && safari-env $* ; }

Collecting names and paths, usage::

   ./precursor.py workflow.bash

Thinking about documenting the precursors via dumping the usage funcs
into a tree of rst  

Issues:

#. run the bash func or parse the source 
#. indentation in the usage
#. timestamp lazy updating
#. generate Sphinx toctree indices with headings in a directory tree mirroring source 
#. sphinx cross referencing, how to do it ? ref or doc
#. where to put the derived files 

    #. NOT directly beside sources ? would introduce updating derived errors and mess up svn status


Alternative more manual approach:

#. pull in usage text via manual use of sphinx extension

    #. avoids two tree problem, the manual tree needs to be in command as the code tree often needs changing 
    #. the usage texts are often pseudo-tracwikitext anyhow

Over the top autodoc style approach:
	
#. Sphinx domain for my flavor of bash functions

    #. despite the snazzyness, this still suffers from two trees issue somewhat  


"""
from __future__ import with_statement
import logging
log = logging.getLogger(__name__)
import re

ptn = re.compile("/(?P<path>\S*)\.bash")

class Precursors(dict):
    def __init__(self, path):	
        dict.__init__(self)
        self.find(path)

    def find(self, path):
        with open(path) as fp:
            for line in filter(lambda _:_[0] != '#',fp.readlines()):
                pos = line.find("-(){") 
                if pos > -1:
                    name = line[:pos]
	            m = ptn.search(line[pos:])	
	            if m:
		        path = m.groupdict()['path']
		        print name, path
			self[name] = path
                    else:		
	                log.warn("for %s failed to match  %s " % (name, line))

    def ordered(self):
        return sorted(self, key=lambda _:self[_])


def opts_():
    from optparse import OptionParser
    op = OptionParser(usage=__doc__)
    defaults=dict(loglevel="INFO")
    op.add_option("-l", "--loglevel",   help="logging level : INFO, WARN, DEBUG ... Default %(loglevel)s " % defaults )
    op.set_defaults( **defaults )
    return op.parse_args()


if __name__ == '__main__':
    opts, args = opts_()
    logging.basicConfig(level=getattr(logging,opts.loglevel.upper()))
    pr = Precursors(args[0])
    for k in pr.ordered():
	print k, pr[k]    

    print pr

