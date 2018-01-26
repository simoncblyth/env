#!/usr/bin/env python
"""
INI FILE CONFIG ONTO BASH COMMANDLINE WITHOUT BASH PARSING HEADACHES
======================================================================

::

    simon:tools blyth$ sect=fossil ; eval $($(env-home)/tools/cnf.py -s $sect) ; _cfg_$sect ; type _cfg_$sect
    _cfg_fossil is a function
    _cfg_fossil () 
    { 
        cnfpath=~/.env.cnf;
        format=bashfn;
        loglevel=INFO;
        binpath=/usr/local/env/fossil/fossil-src-20130216000435/build/fossil;
        sect=fossil;
        repodir=/var/scm/fossil;
        sections=('heprez_trac' 'dybsvn_trac' 'dqtests' 'env_trac' 'fossil' 'workflow_trac' 'shiftcheck');
        port=591;
        user=root
    }
    simon:tools blyth$ 
    simon:tools blyth$ sect=env_trac ; eval $(./cnf.py -s $sect) ; _cfg_$sect ; type _cfg_$sect
    _cfg_env_trac is a function
    _cfg_env_trac () 
    { 
        cnfpath=~/.env.cnf;
        xmlrpc_url=http://blyth:***@dayabay.phys.ntu.edu.tw/tracs/env/login/xmlrpc;
        sect=env_trac;
        format=bashfn;
        loglevel=INFO;
        sections=('heprez_trac' 'dybsvn_trac' 'dqtests' 'env_trac' 'fossil' 'workflow_trac' 'shiftcheck')
    }


Less polluting generated funcs::

    simon:tools blyth$ echo $port
    591
    simon:tools blyth$ n=port
    simon:tools blyth$ echo ${!n}
    591


"""
import os, logging
from pprint import pformat
from ConfigParser import RawConfigParser
RawConfigParser.optionxform = str    # case sensitive keys 

log = logging.getLogger(__name__)

class Cnf(RawConfigParser):
    """
    Simple enhancements to RawConfigParser

    #. expand user and envvar in paths
    #. use case sensitive keys
    #. dictionary based interface

    """
    def read(self, paths ):
        if isinstance(paths, basestring):
            paths = [paths]
        filenames = []    
        for path in paths:
            filenames.append(os.path.expandvars(os.path.expanduser(path)))
        return RawConfigParser.read(self,filenames)

    def sectiondict(self, sect):
        d = {}
        if self.has_section(sect):
            for key in self.options(sect):    
                d[key] = self.get(sect,key)
        return d

    def asdict(self):
        d = {}
        for sect in self.sections():
            d[sect] = self.sectiondict(sect)
        return d    


class BashFn(dict):
    """
    Translate this config section dict into polluting(no local) bash function form::

        _cfg_fossil(){   

        cnfpath=~/.env.cnf
        format=bashfn
        loglevel=INFO
        binpath=/usr/local/env/fossil/fossil-src-20130216000435/build/fossil
        sect=fossil
        repodir=/var/scm/fossil
        sections=('heprez_trac' 'dybsvn_trac' 'dqtests' 'env_trac' 'fossil' 'workflow_trac' 'shiftcheck')
        port=591
        user=root
        }

    Actually as bash looses the newlines have to explicity join with semicolons::

        simon:tools blyth$ fntext=$(cnf.py -s fossil)
        simon:tools blyth$ echo $fntext
        _cfg_fossil(){ cnfpath=~/.env.cnf format=bashfn loglevel=INFO binpath=/usr/local/env/fossil/fossil-src-20130216000435/build/fossil sect=fossil repodir=/var/scm/fossil sections=('heprez_trac' 'dybsvn_trac' 'dqtests' 'env_trac' 'fossil' 'workflow_trac' 'shiftcheck') port=591 user=root }

    Enabling::

        eval $(cnf.py -s fossil) ; _cfg_fossil
        sect=fossil ; eval $(cnf.py -s $sect) ; _cfg_$sect ; type _cfg_$sect



    This allows the config values from the section to be provided onto the bash commandline
    with minimum fuss, and avoids the pain of bash config parsing node dependent flakiness 
    """
    head_ = r"""
_cfg_%(sect)s(){   

"""
    tail_ = r"""
}
"""
    def body(self):
        def kv(_):
            k,v = _
            if type(v) == list:
                v = str(tuple(v)).replace(",","")
            return "%s=%s" % (k,v) 
        return " ; ".join(map(kv, self.items()))
    def __str__(self):
        return self.head_ % self + self.body() + " ; " + self.tail_ % self

def parse_args(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath",   default="~/.env.cnf", help="path to config file Default %default"  )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--format",   default="bashfn", help="output format ... Default %default"  )
    op.add_option("-s", "--sect",      default=None, help="section of config file... Default %default"  )
    op.add_option("--sections", action="store_true", default=False, help="include list of all sections in the config file... Default %default"  )

    opts, args = op.parse_args()
    loglevel = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig(level=loglevel)
    cnf = Cnf()
    cnf.read(opts.cnfpath)

    d = cnf.sectiondict(opts.sect)
    d['sect'] = opts.sect
    if opts.sections: 
        d['sections'] = cnf.sections()
    pass
    d.update(vars(opts))

    log.debug("reading config from sect %s of %s :\n%s " % (opts.sect, opts.cnfpath, pformat(d) ))  
    return opts, args, d

def main():
    opts, args, d = parse_args(__doc__)
    log.debug(pformat(d))
    print str(BashFn(d))  # the only print everything else goes to stderr

if __name__ == '__main__':
    main()

