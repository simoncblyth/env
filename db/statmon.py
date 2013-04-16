#!/usr/bin/env python
"""
File Monitoring
================

Record stat info for a file and the RC from running a command 
with the path as its last argument into an SQLite DB

Objective
----------

Allowing for example to watch a logfile an provide notification
of abnormalities such as 

#. not updating
#. unexpected growth 
#. particular contained strings

Example
--------

Configuring path to monitor and content check command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Config section::

    [profilemon]

    dbpath = ~/.env/profilemon.sqlite
    tn = profile
    orderfield = mtime

    path = ~/.bash_profile
    #cmd = ~/env/db/logcheck.py
    cmd = ls 
    cmd_note = the path to be monitored is appended to the above configured cmd string

    email = blyth@hep1.phys.ntu.edu.tw


Cron config
~~~~~~~~~~~




TODO
=====

#. collect ingredients into self contained package


"""
import os, logging
from pprint import pformat
from datetime import datetime
log = logging.getLogger(__name__)
from ConfigParser import ConfigParser
from simtab import Table
from env.tools.sendmail import notify

statk = "mode ino dev nlink uid gid size atime mtime ctime".split()
stat_ = lambda path:dict(zip(statk,os.stat(os.path.expandvars(os.path.expanduser(path)))))   # dict with stat information
expand_ = lambda path:os.path.expanduser(os.path.expandvars(path))

class StatMon(object):
    tfmt = "%Y-%m-%dT%H:%M:%S"
    def __init__(self, cnf ):
        """
        Establishes connection to configured SQLite DB table, creating
        the DB and the table if they do not exist

        Hmm, table definition from cnf too ? Probably not so useful
        as collection of the data tied to its nature

        :param cnf:
        """

        kwa = {}
        kwa['rc'] = "int"   # exit code from the logfile checking script
        for k in statk:
            kwa[k] = "int"   # spell it out for py23 

        self.cnf = cnf
        self.tab = Table(cnf['dbpath'], cnf['tn'], **kwa )

    def rec(self, cmd):
        """
        Run the command in a separate process, record the 
        return code from running the configured command together
        with the results of stating the file prior to running the commnd

        :param cmd:
        """
        log.info("running cmd %s " % cmd)
       
        path = expand_(self.cnf['path'])
        assert os.path.exists(path), "path %s " % path 

        rec = stat_(path)

        pipe = os.popen(cmd)
        ret = pipe.read().strip()
        status = pipe.close()
        if not status:
            status = 0
        rec['rc'] = os.WEXITSTATUS(status)

        log.info("rec %s " % (rec)) 
        self.tab.add( **rec ) 
        self.tab.insert()

    def ls(self):
        """
        Dump all entried in the configured table
        """
        sql = "select * from %(tn)s " % self.cnf
        for d in self.tab(sql):
            print d

    def rep(self):
        """
        Use sqlite3 binary to present the last few entries in the configured table
        """
        return os.popen("echo \"select * from %(tn)s order by %(orderfield)s desc limit 24 ;\" | sqlite3 %(dbpath)s " % self.cnf).read() 

    def mon(self):
        """
        Examine last entry in the configured DB table, and compares value to configured constraints
        When excursions are seen send notification email, to configured email addresses
        """
        last = self.tab.iterdict("select * from %(tn)s order by %(orderfield)s desc limit 1" % self.cnf).next()

        for k in ('mtime','ctime','atime'):
            t = datetime.fromtimestamp(last[k])
            last[k] = t.strftime(self.tfmt)

        excursion = False
        if excursion: 
            subj = "WARNING excursion for entry mtime %(mtime)s " 
        else: 
            subj = "OK last entry : mtime %(mtime)s  "

        subj = subj % last 
        if excursion:
            msg = "\n".join([subj, self.rep()]) 
            log.warn("sending notification as: %s " % subj )
            #notify(self.cnf['email'], msg )
        else:
            log.info("no notification: %s " % subj )
        pass

    def _cmd(self):
         """
         :return: configured command with the file to be monitored appended
         """
         return "%s %s" % ( expand_(self.cnf['cmd']), self.cnf['path'] )

    def __call__(self, args):
        """
        Performs action for each argument string

        :param args: list of argument strings
        """
        for arg in args:
            log.info("arg %s" % arg )
            if arg == 'rec':
                self.rec(self._cmd())
            elif arg == 'ls':
                self.ls()
            elif arg == 'mon':
                self.mon()
            elif arg == 'rep':
                print self.rep()
            else:
                log.warn("unhandled arg %s " % arg ) 


class Cnf(dict):
    expect = [
        ("orderfield", "mtime",           "Field to use to entry ordering", ),
        ("path",       "~/.bash_profile", "Filepath to monitor", ),
    ]
    def __init__(self, sect, cnfpath ):
        """
        Read `sect` section of config file into this dict and 

        :param sect: section name in 
        :param cnfpath: config file path
        """
        cpr = ConfigParser()
        cpr.read(os.path.expanduser(cnfpath))
        for k,v in cpr.items(sect):# spell it out for py2.3
            self[k] = v 
        self['sect'] = sect
        self['cnfpath'] = cnfpath
        self['sections'] = cpr.sections()

    def check(self):
        for _ in self.expect:
            k, example, msg = _
            assert k in self, "Missing expected config key \"%s\" from %s : eg %s : %s " % ( k, repr(self), example, msg  )  


def parse_args(doc):
    """
    Return config dict and commandline arguments 

    :param doc:
    :return: cnf, args  
    """
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath",    default="~/.env.cnf", help="path to config file Default %default"  )
    op.add_option("-s", "--sect",       default="profilemon", help="section of config file... Default %default"  )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )

    opts, args = op.parse_args()
    loglevel = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig()   # for py2.3 compatibility
    logging.getLogger().setLevel(loglevel)

    cnf = Cnf(opts.sect, opts.cnfpath)
    cnf.check()
    log.debug("reading config from sect %s of %s :\n%s " % (opts.sect, opts.cnfpath, cnf))  
    return cnf, args


if __name__ == '__main__':
    cnf, args = parse_args(__doc__)
    print pformat(cnf)
    StatMon(cnf)(args)


