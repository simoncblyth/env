#!/usr/bin/env python
"""

"""
import logging, os
from ConfigParser import RawConfigParser 
from optparse import OptionParser
log = logging.getLogger(__name__)

class Cnf(dict):pass

def cnf_(doc):
    """
    Config from `--cnfpath` is potentially overridden by commandline settings

    :param doc: docstring for commandline help message 
    :return: `Cnf` instance holding config from file and commandline

    """ 
    parser = OptionParser(doc)
    parser.add_option("-c", "--cnfpath", help="file from which to load config setting.", default="~/.env.cnf" )
    parser.add_option("-l", "--level", help="loglevel", default="INFO" )
    parser.add_option("-n", "--npull", type=int, help="restrict retreival to first n links", default=10 )
    parser.add_option("-o", "--outd", default=None, help="directory in which to output retreived files, the default is based on configured base and the target url")
    parser.add_option("-s", "--site", type=str, help="default site if no argument supplied", default="shiftcheck" )
    parser.add_option("--only", help="string with comma delimited list of item names restricting operations", default="" )

    (opts, args) = parser.parse_args()
    logging.basicConfig(level=getattr(logging,opts.level.upper()))

    #assert len(args) == 1, "must supply siteconf section name present in %s " % opts.cnfpath
    cpr=RawConfigParser()
    site = args[0] if len(args)>0 else opts.site
    cpr.read(os.path.expanduser(opts.cnfpath))
    d = Cnf(cpr.items(site))
    d.site = site
    d.outd = opts.outd
    d.npull = opts.npull
    d.only = opts.only 
    return d

if __name__ == '__main__':
    cnf = cnf_(__doc__)
    print cnf
