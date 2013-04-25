#!/usr/bin/env python
"""
File Monitoring
================

Record stat info for a file and the RC from running a command 
with the path as its last argument into an SQLite DB

Objective
----------

Allowing for example to watch a logfile and provide notification
of abnormalities such as 

#. not updating
#. unexpected growth 
#. particular contained strings

Usage
------

::

    ~/env/db/statmon.py -s dybslvmon ls rec mon rep

Command strings can be used singly or together:

rec 
    capture the stat of the configured path and RC of configured command

mon
    apply monitoring checks to the entries in the DB, potentially sending notification
    emails when expectations are violated

ls
    dump entries in DB table as dicts

rep
    sqlite3 table presentation 



Examples
--------

Configuring path to monitor and content check command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Config section::

    [profilemon]

    path = ~/.bash_profile
    dbpath = ~/.env/profilemon.sqlite
    tn = profile
    orderfield = ltime
    lookback = days:7  
    cmd = ls %%(path)s
    email = blyth@hep1.phys.ntu.edu.tw

    [dybslvmon]

    path = /data1/env/local/dyb/dybinst-slave.log
    dbpath = ~/.env/dybslvmon.sqlite
    tn = dybslv
    orderfield = ltime
    lookback = days:7  
    cmd = ~/env/db/dybslvmon.py %%(path)s
    email = blyth@hep1.phys.ntu.edu.tw


Cron config
~~~~~~~~~~~

::

   35 * * * * ( cd $ENV_HOME ; $ENV_HOME/db/statmon.py -s dybslvmon dybinst )  > $CRONLOG_DIR/dybslvmon_.log 2>&1



TODO
=====

#. collect ingredients into self contained package


"""
import os, logging, time
from pprint import pformat
from datetime import datetime, timedelta
log = logging.getLogger(__name__)
from ConfigParser import ConfigParser
from simtab import Table
from env.tools.sendmail import notify

statk = "mode ino dev nlink uid gid size atime mtime ctime".split()
stat_ = lambda path:dict(zip(statk,os.stat(os.path.expandvars(os.path.expanduser(path)))))   # dict with stat information
expand_ = lambda path:os.path.expanduser(os.path.expandvars(path))
tfmt_ = lambda _:_.strftime("%Y-%m-%d %H:%M:%S")

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
        kwa['rc'] = "int"      # exit code from the logfile checking script
        kwa['ltime'] = "int"   # look time
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
        rec['ltime'] = int(time.time()) 

        log.info("rec %s " % (rec)) 
        self.tab.add( **rec ) 
        self.tab.insert()

    def ls(self):
        """
        Dump all entries in the configured table
        """
        sql = "select * from %(tn)s " % self.cnf
        for d in self.tab(sql):
            print d

    def growth(self):
        """
        Are any periods of flatlining logsize OK ? 
        Would not think so as slave tries to talk to master every 5 mins.::

		2013-04-24 01:15:01 1023171347 13392
		2013-04-24 01:30:01 1023179602 8255
		2013-04-24 01:45:01 1023192331 12729
		2013-04-24 02:00:01 1023196560 4229
		2013-04-24 02:15:01 1023196560 0
		2013-04-24 02:30:01 1023196560 0
		2013-04-24 02:45:01 1023196560 0
		2013-04-24 03:00:01 1023196560 0
		2013-04-24 03:15:01 1023196560 0
		2013-04-24 03:30:01 1023196560 0
		2013-04-24 03:45:01 1023196560 0
		2013-04-24 04:00:02 1023196560 0
		2013-04-24 04:15:02 1023196560 0
		2013-04-24 04:30:01 1023196560 0
		2013-04-24 04:45:01 1023209322 12762
		2013-04-24 05:00:02 1023209972 650
		2013-04-24 05:15:01 1023211901 1929
		2013-04-24 05:30:01 1023211901 0
		2013-04-24 05:45:01 1023220203 8302
		2013-04-24 06:00:01 1023258033 37830
		2013-04-24 06:15:01 1023270669 12636   

                      corresponds to end of a spate of grep urlopen\ error dybinst-slave.log  erros
                      followed by slave getting stuck at init_sockobject lockup
                         http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/293 

		2013-04-24 06:30:02 1023270669 0
		2013-04-24 06:45:01 1023270669 0
		2013-04-24 07:00:01 1023270669 0
		2013-04-24 07:15:01 1023270669 0
		2013-04-24 07:30:01 1023270669 0
		2013-04-24 07:45:02 1023270669 0
		2013-04-24 08:00:01 1023270669 0
		2013-04-24 08:15:01 1023270669 0
		2013-04-24 08:30:01 1023270669 0
		2013-04-24 08:45:01 1023270669 0
		2013-04-24 09:00:01 1023270669 0
		2013-04-24 09:15:02 1023270669 0
		2013-04-24 09:30:01 1023270669 0
		2013-04-24 09:45:01 1023270669 0
		2013-04-24 10:00:01 1023270669 0

         During timeconsuming steps like::

		2013-04-25 13:37:41,580 [bitten.build.shtools] INFO:     Configuring rest of dybgaudi ...done
		2013-04-25 14:47:04,920 [bitten.build.shtools] INFO:     Building dybgaudi ...done
		2013-04-25 14:47:05,121 [bitten.build.api] DEBUG: bash exited with code 0
		2013-04-25 14:47:05,121 [bitten.slave] INFO: Build step dybgaudi completed successfully

         Growth flatlined for an hour::

		2013-04-25 13:30:01 1023270669 0
		2013-04-25 13:45:01 1023304204 33535
		2013-04-25 14:00:01 1023304204 0
		2013-04-25 14:15:01 1023304204 0
		2013-04-25 14:30:02 1023304204 0
		2013-04-25 14:45:01 1023304204 0
		2013-04-25 15:00:01 1023317590 13386

        """
        sql = "select size, ltime from %(tn)s where ltime > %(tzero)s order by ltime" % self.cnf
        psize = 0
        for d in self.tab.listdict(sql, fields="size,ltime"):
            print datetime.fromtimestamp(d['ltime']), d['size'], d['size'] - psize
            psize = d['size']


    def rep(self):
        """
        Use sqlite3 binary to present the last few entries in the configured table
        ::
		 
		ctime                mtime                atime                size        rc          nlink     
		-------------------  -------------------  -------------------  ----------  ----------  ----------
		2013-04-22 13:20:00  2013-04-22 13:20:00  2013-04-16 18:24:12  1021700529  0           1         
		2013-04-22 13:20:00  2013-04-22 13:20:00  2013-04-16 18:24:12  1021700529  0           1         
		2013-04-22 13:30:46  2013-04-22 13:30:46  2013-04-16 18:24:12  1021702521  0           1         
		2013-04-22 13:30:46  2013-04-22 13:30:46  2013-04-16 18:24:12  1021702521  0           1         
		2013-04-22 13:30:46  2013-04-22 13:30:46  2013-04-16 18:24:12  1021702521  0           1         

        """
        dt_ = lambda f:"datetime(%s,'unixepoch','localtime') as %s" % (f,f) 
        fields = ",".join( map(dt_,("ltime","ctime","mtime","atime")) + "size nlink rc".split() ) 
        sql = "select %(fields)s from %(tn)s order by %(orderfield)s desc limit %(limit)s " % dict(self.cnf, fields=fields, limit=10 )
        log.info(sql)
        return os.popen("echo \"" + sql + " ;\" | sqlite3 %(dbpath)s " % self.cnf).read() 

    def mon(self):
        """
        Examine last entry in the configured DB table, and compares value to configured constraints
        When excursions are seen send notification email, to configured email addresses
        """
        last = self.tab.iterdict("select * from %(tn)s order by %(orderfield)s desc limit 1" % self.cnf).next()

        for k in 'ltime mtime ctime atime'.split():
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
         :return: configured command 
         """
         return expand_(self.cnf['cmd'])

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
            elif arg == 'gro':
                self.growth()
            else:
                log.warn("unhandled arg %s " % arg ) 


class Cnf(dict):
    expect = [
        ("orderfield", "ltime",           "Field to use to entry ordering", ),
        ("lookback",   "days:10",         "string period specification used to create a timedelta", ),
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

def parse_timedelta(s):
    """
    :param s: timedelta string such as "days:7"
    :return: timedelta object representing the period
    """
    l = s.split(":")
    assert len(l) == 2, "expecting string like days:7"
    t = timedelta(**{l[0]:int(l[1])})  # transient dict to allow dynamic keyword 
    return t 

def tzero(lookback_, now=None):
    """
    :param lookback_: string representing a `timedelta` instance 
    :return: unix timestamp corresponding to start of period 
    """
    lookback = parse_timedelta(lookback_)
    if not now:
        now = datetime.now()
    dt0 = now - lookback
    ts0 = int(time.mktime(dt0.timetuple()))  # convert datetime into unix timestamp
    log.info("converted lookback_ %s from now %s into dt0 %s ts0 %s " % (lookback_, tfmt_(now), tfmt_(dt0), ts0 ))
    return ts0

def parse_args(doc):
    """
    Return config dict and commandline arguments 

    :param doc:
    :return: cnf, args  
    """
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath",    default="~/.env.cnf", help="path to config file Default %default"  )
    op.add_option("-s", "--sect",       default="dybslvmon", help="section of config file... Default %default"  )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )

    opts, args = op.parse_args()
    loglevel = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig()   # for py2.3 compatibility
    logging.getLogger().setLevel(loglevel)

    cnf = Cnf(opts.sect, opts.cnfpath)
    cnf.check()
    cnf['tzero'] = tzero(cnf['lookback'])
    log.debug("reading config from sect %s of %s :\n%s " % (opts.sect, opts.cnfpath, cnf))  
    return cnf, args


def main():
    cnf, args = parse_args(__doc__)
    print pformat(cnf)
    StatMon(cnf)(args)


if __name__ == '__main__':
    main()

