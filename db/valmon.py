#!/usr/bin/env python
"""
Value Monitoring
===================

.. warning:: Keeping this operational with ancient python23 is advantageous

Simple monitoring and recording the output of commands that 
return a single value. The result is stored with a timestamp
in an sqlite DB

Config example::

    [oomon]

    note = despite notification being enabled this failed to notify me, apparently the C2 OOM issue made the machine incapable of sending email ?
    cmd = grep oom /var/log/messages | wc -l  
    constraints = ( val == 0, )
    dbpath = ~/.env/oomon.sqlite
    tn = oomon

    [envmon]

    note = check C2 server from cron on C, 
    cmd = curl -s --connect-timeout 3 http://dayabay.phys.ntu.edu.tw/repos/env/ | grep trunk | wc -l
    valmin = -100
    valmax = 100 
    constraints = ( val == 1 and val < valmax, val > valmin, val < valmax, )
    instruction = 
                  the simple python constraint expression is evaluated within the scope of the other config values 
                  the constraints string needs to evaluate to a tuple of one or more bools

    dbpath = ~/.env/envmon.sqlite
    tn = envmon


Usage::

    $ENV_HOME/db/valmon.py -s oomon rec ls rep mon

Crontab::

    #50 * * * * ( export HOME=/root ; LD_LIBRARY_PATH=/data/env/system/python/Python-2.5.6/lib /data/env/system/python/Python-2.5.6/bin/python /home/blyth/env/db/valmon.py -s oomon rec mon ; ) > /var/scm/log/oomon.log 2>&1
	50 * * * * ( export HOME=/root ; /home/blyth/env/db/valmon.py -s oomon rec mon ; ) > /var/scm/log/oomon.log 2>&1

On C2, was forced to use source rather than system python 2.3 until `yum installed python-sqlite2`, see simtab for notes on this.

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
        """
        Establishes connection to configured SQLite DB table, creating
        the DB and the table if they do not exist

        Hmm, table definition from cnf too ? Probably not so useful
        as collection of the data tied to its nature

        :param cnf:
        """
        self.cnf = cnf
        self.tab = Table(cnf['dbpath'], cnf['tn'], date="text", val="real" )

    def interpret_as_int(self, ret):
        """
        :param ret: string returned by a command
        :return: integer or None if the string cannot be coerced into an integer
        """
        try:
           val = typ(ret)
        except ValueError:
           log.warn("non integer returned by cmd %s " % ret )
           val = None
        return val   

    def float_or_asis(self, arg):
        """
        :param arg: arg that is possibly representing a float
        :return: float from the arg OR unchanged arg if could not be coerced
        """
        try:
           v = float(arg)
        except ValueError:
           v = arg
        except TypeError:
           v = arg
        return v 


    def rec(self, cmd):
        """
        Run the command in a separate process, interpret the
        response as an integer and save and a timestamp 
        it into the configured DB table

        :param cmd:
        """
        log.info("running cmd %s " % cmd)
        ret = os.popen(cmd).read().strip()
        val = self.interpret_as_int(ret)
        log.info("ret %s val %s " % (ret, val )) 
        if val != None:
            dt = datetime.now()
            self.tab.add( val=val, date=dt.strftime("%Y-%m-%dT%H:%M:%S") ) 
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
        return os.popen("echo \"select * from %(tn)s order by date desc limit 24 ;\" | sqlite3 %(dbpath)s " % self.cnf).read() 


    def constrain_(self, last):
        """
        :param last: dict of last DB entry 

        #. hmm a general solution would allow constraints specified by configured expression strings  
        
            * need a sanitizing expression parser to operate within the context of the last dict variables  
            * http://effbot.org/zone/simple-top-down-parsing.htm

        BUT i do not need the complexity, just a very simple expression parsing

        Hmm constraining the scope of eval looks attractive::

	   In [55]: eval(" ( val > valmax, val == valmax, val < valmax ,  valmin <= val <= valmax) ", dict(val=70,valmax=99,valmin=50) , {} )
	   Out[55]: (False, False, True, True)

        BUT do not use this in situations where security of the strings is less than python code,
        as the strings must be considered to be python code::

          In [53]: print eval("open('.bash_profile').read()", {},{})  ## succeeds to print 

        """
        ctx = {}
        for k,v in last.items():
            ctx[k] = v
        for k,v in self.cnf.items():
            ctx[k] = self.float_or_asis(v)    

        strip_ = lambda _:_.strip().lstrip()
        constraints = strip_(self.cnf['constraints'])
        assert constraints[0] == '(' and constraints[-1] == ')', ("unexpected constraints", constraints )
        lbls = filter(len,map(strip_,constraints[1:-1].split(",")))
        evls = eval(constraints, ctx, {} )
        del ctx['__builtins__']             # rm messy side effect of the eval  
        assert len(lbls) == len(evls)
        edict = dict(zip(lbls,evls))
        return edict, ctx

    def mon(self):
        """
        Examine last entry in the configured DB table, and compares value to configured constraints
        When excursions are seen send notification email, to configured email addresses
        """
        last = self.tab.iterdict("select * from %(tn)s order by date desc limit 1" % self.cnf).next()
        edict, ctx = self.constrain_(last)
        log.debug("ctx:\n %s " % pformat(ctx))
        log.debug("edict:\n %s " % pformat(edict))

        subj = "last entry from %(date)s" 
        subj = subj % last 
        oks =  map(lambda _:_[0],filter(lambda _:_[1],     edict.items() ))
        exc =  map(lambda _:_[0],filter(lambda _:not _[1], edict.items() ))

        if len(exc) > 0: 
            subj += "WARN: " + "  ****  ".join(exc) 

        if len(oks) > 0:
            subj += "OK: " + "  ____  ".join(oks) 

        if len(exc) > 0:
            msg = "\n".join([subj, self.rep()]) 
            log.warn("sending notification as: %s " % subj )
            notify(self.cnf['email'], msg )
        else:
            log.info("no notification: %s " % subj )


    def __call__(self, args):
        """
        Performs action for each argument string

        :param args: list of argument strings
        """
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
        """
        Read `sect` section of config file into this dict

        :param sect: section name in 
        :param cnfpath: config file path
        """
        cpr = ConfigParser()
        cpr.read(os.path.expanduser(cnfpath))
        for k,v in cpr.items(sect):# spell it out for py2.3
            self[k] = v 
        self['sect'] = sect
        self['sections'] = cpr.sections()


def parse_args(doc):
    """
    Return config dict and commandline arguments 

    :param doc:
    :return: cnf, args  
    """
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



