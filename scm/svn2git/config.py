#!/usr/bin/env python
"""

"""
from ConfigParser import ConfigParser
import os, logging
log = logging.getLogger(__name__)

class Cnf(dict):
    def __init__(self, sect, cnfpath="~/.env.cnf" ):
        cpr = ConfigParser()
        cpr.read(os.path.expanduser(cnfpath))
        self.update(cpr.items(sect)) 
        self['sect'] = sect
        self['sections'] = cpr.sections()

def parse_args(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath",   default="~/.env.cnf", help="path to config file Default %default"  )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-s", "--sect",      default="svnauthors", help="section of config file... Default %default"  )
    opts, args = op.parse_args()
    loglevel = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig(level=loglevel)
    cnf = Cnf(opts.sect, opts.cnfpath)
    log.debug("reading config from sect %s of %s :\n%s " % (opts.sect, opts.cnfpath, cnf))  
    return cnf, args

if __name__ == '__main__':
    cnf, args = parse_args(__doc__)
    print cnf
    print args 
