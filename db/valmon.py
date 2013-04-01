#!/usr/bin/env python
"""
Simple monitoring and recording the output of commands that 
return a single value. The result is stored with a timestamp
in an sqlite DB

Config::

    [oomon]

    cmd = grep oom /var/log/messages | wc -l  
    dbpath = ~/.env/oomon.sqlite
    tn = oomon

Usage::

    $ENV_HOME/db/valmon.py -s oomon


"""
import os, logging
from pprint import pformat
from datetime import datetime
log = logging.getLogger(__name__)
from ConfigParser import ConfigParser
from simtab import Table


class ValueMon(object):
    def __init__(self, cnf ):
        self.cnf = cnf
        self.tab = Table(cnf['dbpath'], cnf['tn'], date="text", val="real" )

    def interpret_as_int(self, ret):
        try:
           val = int(ret)
        except ValueError:
           log.warn("non integer returned by cmd %s " % ret )
           val = None
        return val   

    def rec(self, cmd):
        log.info("running cmd %s " % cmd)
        ret = os.popen(cmd).read().strip()
        val = self.interpret_as_int(ret)
        log.info("ret %s val %s " % (ret, val )) 
        if val != None:
            dt = datetime.now()
            self.tab.add( val=val, date=dt.strftime("%Y-%m-%dT%H:%M:%S") ) 
            self.tab.insert()

    def ls(self):
        sql = "select * from %(tn)s " % self.cnf
        for d in self.tab(sql):
            print d

    def __call__(self, args):
        for arg in args:
            log.info("arg %s" % arg )
            if arg == 'rec':
                self.rec(self.cnf['cmd'])
            elif arg == 'ls':
                self.ls()
            else:
                log.warn("unhandled arg %s " % arg ) 


class Cnf(dict):
    def __init__(self, sect, cnfpath="~/.env.cnf" ):
        cpr = ConfigParser()
        cpr.read(os.path.expanduser(cnfpath))
        for k,v in cpr.items(sect):# spell it out for py2.3
            self[k] = v 
        self['sect'] = sect
        self['sections'] = cpr.sections()


def parse_args(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath",   default="~/.env.cnf", help="path to config file Default %default"  )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-s", "--sect",      default="oomon", help="section of config file... Default %default"  )
    opts, args = op.parse_args()
    loglevel = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig()   # for py2.3 compatibility
    logging.getLogger().setLevel(loglevel)

    cnf = Cnf(opts.sect, opts.cnfpath)
    log.debug("reading config from sect %s of %s :\n%s " % (opts.sect, opts.cnfpath, cnf))  
    return cnf, args


if __name__ == '__main__':
    cnf, args = parse_args(__doc__)
    ValueMon(cnf)(args)



