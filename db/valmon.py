#!/usr/bin/env python
"""
Simple monitoring and recording the output of commands that 
return a single value. The result is stored with a timestamp
in an sqlite DB

Config::

    [oomon]

    cmd = grep oom /var/log/messages | wc -l  
    valmax = 0
    dbpath = ~/.env/oomon.sqlite
    tn = oomon

Usage::

    $ENV_HOME/db/valmon.py -s oomon rec ls rep mon

Crontab::

        #50 * * * * ( export HOME=/root ; LD_LIBRARY_PATH=/data/env/system/python/Python-2.5.6/lib /data/env/system/python/Python-2.5.6/bin/python /home/blyth/env/db/valmon.py -s oomon rec mon ; ) > /var/scm/log/oomon.log 2>&1
	50 * * * * ( export HOME=/root ; /home/blyth/env/db/valmon.py -s oomon rec mon ; ) > /var/scm/log/oomon.log 2>&1

On C2, was forced to use source rather than system python 2.3 until yum installed python-sqlite2, see simtab for notes on this.

"""
import os, logging
from pprint import pformat
from datetime import datetime
log = logging.getLogger(__name__)
from ConfigParser import ConfigParser
from simtab import Table
from env.tools.sendmail import notify


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

    def rep(self):
        return os.popen("echo \"select * from %(tn)s order by date desc limit 24 ;\" | sqlite3 %(dbpath)s " % self.cnf).read() 

    def mon(self):
        last = self.tab.iterdict("select * from %(tn)s order by date desc limit 1" % self.cnf).next()
        last['valmax'] = float(self.cnf['valmax'])
        val_high = last['val'] > last['valmax']
        if val_high: 
            subj = "WARNING last entry from %(date)s,  val %(val)s > max %(valmax)s " 
        else: 
            subj = "OK last entry from %(date)s, val %(val)s < max %(valmax)s "
        subj = subj % last 
        if val_high:
            msg = "\n".join([subj, self.rep()]) 
            log.warn("sending notification as: %s " % subj )
            notify(self.cnf['email'], msg )
        else:
            log.info("no notification: %s " % subj )


    def __call__(self, args):
        for arg in args:
            log.info("arg %s" % arg )
            if arg == 'rec':
                self.rec(self.cnf['cmd'])
            elif arg == 'ls':
                self.ls()
            elif arg == 'mon':
                self.mon()
            elif arg == 'rep':
                print self.rep()
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



