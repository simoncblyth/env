#!/usr/bin/env python
"""
cnf.py
=======


Bash Access to value of single config key 
-------------------------------------------

Note this allows config values to reside in the 
ini file and be easily accessed from bash or python

::

   wtracdb-cnf(){   ~/env/web/cnf.py  -c ~/.workflow.cnf -s workflow_trac2sphinx -k ${1:-sphinxdir} ; }


"""
import logging, os
from ConfigParser import RawConfigParser 
from optparse import OptionParser
log = logging.getLogger(__name__)

class Cnf(dict):
    @classmethod 
    def read(cls, sect, path="~/.env.cnf"):
        cpr=RawConfigParser()
        cpr.read(os.path.expanduser(path))
        d = cls(cpr.items(sect))
        d['cnfsect'] = sect
        d['cnfpath'] = path
        return d 

    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)

    def __repr__(self):
        return "<Cnf %(cnfsect)s %(cnfpath)s>" % self

    def _get_bash(self):
        sect = self['cnfsect']
        return "\n".join(["%s_%s(){ echo \"%s\" ; } " % (sect, k, self[k]) for k in self if not k.startswith('cnf') ]) 
    bash = property(_get_bash)
  


def cnf_(doc, **kwa):
    """
    Config from `--cnfpath` is potentially overridden by commandline settings

    :param doc: docstring for commandline help message 
    :return: `Cnf` instance holding config from file and commandline

    """ 

    dflt = {}
    dflt["cnfpath"] = kwa.get("cnfpath", "~/.env.cnf" )
    dflt["level"] = kwa.get("level", "INFO" )
    dflt["npull"] = kwa.get("npull", 10 )
    dflt["outd"] = kwa.get("outd", None )
    dflt["site"] = kwa.get("site", "shiftcheck" )

    parser = OptionParser(doc)
    parser.add_option("-c", "--cnfpath", help="file from which to load config setting.", default=dflt["cnfpath"] )
    parser.add_option("-l", "--level", help="loglevel", default=dflt["level"] )
    parser.add_option("-n", "--npull", type=int, help="restrict retreival to first n links", default=dflt["npull"] )
    parser.add_option("-o", "--outd", default=dflt["outd"], help="directory in which to output retreived files, the default is based on configured base and the target url")
    parser.add_option("-s", "--site", type=str, help="default site if no argument supplied", default=dflt["site"] )
    parser.add_option("-k", "--key",   default=None, help="output to stdout just the value of the key"  )
    parser.add_option("--only", help="string with comma delimited list of item names restricting operations", default="" )

    (opts, args) = parser.parse_args()
    logging.basicConfig(level=getattr(logging,opts.level.upper()))

    cnfpath = opts.cnfpath

    site = args[0] if len(args)>0 else opts.site

    d = Cnf.read(site, opts.cnfpath )
    d.outd = opts.outd
    d.npull = opts.npull
    d.only = opts.only 
    d.key = opts.key

    return d



if __name__ == '__main__':
    cnf = cnf_(__doc__)
    if cnf.key is not None:
        print cnf[cnf.key]
    else: 
        print cnf.bash
    pass
