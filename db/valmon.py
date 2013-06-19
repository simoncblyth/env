#!/usr/bin/env python
"""
Value Monitoring
===================

.. warning:: Keeping this operational with ancient python23 is advantageous

Simple monitoring and recording the output of commands that 
return a single value or a dict string. 
The results are stored with a timestamp in a sqlite DB

Usage with `diskmon` section is shown below. 
The section must correspond to a section name in the config file, which defaults to :file:`~/.env.cnf`::

    valmon.py -s diskmon rec rep mon 

Usage from cron::

    52 * * * * ( valmon.py -s diskmon rec rep mon ) > $CRONLOG_DIR/diskmon.log 2>&1 

Installation
-------------

Usage of the `valmon.py` script and the **env** python modules that 
it is based upon requires these to be installed as described at :doc:`/install`.
Essentially this just requires a symbolic link from python `site-packages` and 
a PATH setting to give easy access to scripts from eg `/root/env/bin`

Separation of concerns
-----------------------

The value monitoring in `valmon.py` is kept generic, with all the specifics 
of obtaining the values handled within the command called and choosing constraints 
to apply to them within the config.

For example the `diskmon` section uses the `disk_usage.py` script which returns a dict string::

    [blyth@cms01 e]$ disk_usage.py 
    {'gb_total': '131.74', 'gb_free': '24.90', 'percent_free': '18.90', 'percent_used': '76.02'}

Other sections like `oomon` monitors the single integer returned by the below command:: 

    [root@cms02 ~]# grep oom /var/log/messages | wc -l 
    0 

This approach allows the value monitoring and persistence framework to be reused for
monitoring any quantity which commands or scripts can be written to obtain.


Command arguments
-------------------

`rec`
     record status into the SQLite DB, but running the configured command and storing results 
`mon` 
     check if the last entry in the DB table conforms to the expectations, if not set notification email 
`rep`
     status report based on the DB entries
`msg`
     show what the notification email and subject would be without sending email, 
     a blank msg indicates that no email would be sent
`ls`
     a simple query against the table for the configured section, for debugging


Schema Versions
----------------

`0.1`
     `date`, `val` 
`0.2`
     `date`, `val`, `runtime`, `rc`, `ret`


Configuration
--------------

The command to run and the constraints applied to what it returns 
are obtained from config.  This approach is taken to allow most typical 
changes of varying constraints to be done via configuration only. 

Examples::

    [oomon]

    note = despite notification being enabled this failed to notify me, apparently the C2 OOM issue made the machine incapable of sending email ?
    cmd = grep oom /var/log/messages | wc -l  
    return = int
    constraints = ( val == 0, )
    dbpath = ~/.env/oomon.sqlite
    tn = oomon

    [diskmon]

    note = stores the dict returned by the command as a string in the DB without interpretation
    cmd = disk_usage.py /data
    valmon_version = 0.2 
    return = dict
    constraints = ( gb_free > 10, )
    dbpath = ~/.env/envmon.sqlite
    tn = diskmon

    [dbsrvmon]

    note = currently set to fail via age
    chdir = /var/dbbackup/dbsrv/belle7.nuu.edu.tw/channelquality_db_belle7/archive/10000
    cmd = digestpath.py 
    valmon_version = 0.2 
    return = dict
    constraints = ( tarball_count >= 34, dna_mismatch == 0, age < 86400 , age < 1000, )
    dbpath = ~/.env/dbsrvmon.sqlite
    tn = channelquality_db


    [envmon]

    note = check C2 server from cron on other nodes
    hostport = dayabay.phys.ntu.edu.tw
    # from N need to get to C2 via nginx reverse proxy on H
    #hostport = hfag.phys.ntu.edu.tw:90  
    cmd = curl -s --connect-timeout 3 http://%(hostport)s/repos/env/ | grep trunk | wc -l
    return = int
    constraints = ( val == 1, )
    instruction = require a single trunk to be found, verifying that the apache interface to SVN is working 
    observations = may 16, 2013 observing variable response times that triggering notifications with a 3s timeout    
    dbpath = ~/.env/envmon.sqlite
    tn = envmon

    [envmon_demo]

    note = check C2 server from cron on C, 
    cmd = curl -s --connect-timeout 3 http://dayabay.phys.ntu.edu.tw/repos/env/ | grep trunk | wc -l
    return = int
    valmin = -100
    valmax = 100 
    constraints = ( val == 1 and val < valmax, val > valmin , val < valmax )
    instruction = 
        the simple python `constraints` expression is evaluated within the scope of 
        the section config values (with things that can be coerced to floats so coerced)
        the constraint needs to evaluate to a tuple of one or more bools. 
        To specify a one element tuple a trailing comma is needed, eg "( val > valmin, )"

    dbpath = ~/.env/envmon.sqlite
    tn = envmon


Source python cron
~~~~~~~~~~~~~~~~~~~

When forced to use source rather than system python 2.3 on C2 had to 
setup the cron environment accordingly::

     SHELL=/bin/bash
     HOME=/home/blyth
     ENV_HOME=/home/blyth/env
     CRONLOG_DIR=/home/blyth/cronlog
     PATH=/home/blyth/env/bin:/data/env/system/python/Python-2.5.1/bin:/usr/bin:/bin
     LD_LIBRARY_PATH=/data/env/system/python/Python-2.5.1/lib
     42 * * * * * ( valmon.py -s envmon rec rep mon ) > $CRONLOG_DIR/envmon.log 2>&1 

Avoided this complication by `yum install python-sqlite2`, see simtab for notes on this.


"""
import os, sys, logging, time, platform, pwd
from pprint import pformat
from datetime import datetime
log = logging.getLogger(__name__)
from ConfigParser import ConfigParser
from simtab import Table
from env.tools.sendmail import notify

def excepthook(*args):
    log.error('Uncaught exception:', exc_info=args)
sys.excepthook = excepthook # without this assert tracebacks and messages do not make it into the log 


class ValueMon(object):
    def __init__(self, cnf ):
        """
        Establishes connection to configured SQLite DB table, creating
        the DB and the table if they do not exist

        Hmm, table definition from cnf too ? Probably not so useful
        as collection of the data tied to its nature

        :param cnf:
        """
        log.debug(str(cnf))
        version = cnf.get('valmon_version','0.1')
        if version == '0.1':
            tab = Table(cnf['dbpath'], cnf['tn'], date="text", val="real")
        elif version == '0.2':
            tab = Table(cnf['dbpath'], cnf['tn'], date="text", val="real", runtime="real", rc="int", ret="text" )
        else:
            assert 0, ("unexpected valmon_version %s " % version, cnf)  
        pass
        self.cnf = cnf
        self.version = version
        self.tab = tab

    def interpret_as_int(self, ret):
        """
        :param ret: string returned by a command
        :return: integer or None if the string cannot be coerced into an integer
        """
        try:
           val = int(ret)
        except ValueError:
           log.warn("non integer returned by cmd %s " % ret )
           val = None
        return val   

    def interpret_as_dict(self, ret):
        """
        :param ret: string returned by a command
        :return: dict 
        """
        s = ret.lstrip().strip()
        assert s[0] == '{' and s[-1] == '}', "doesnt look like a dict >>>%s<<< " % s 
        d = eval(s)
        return d

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

        chdir = self.cnf.get('chdir',None) 
        if not chdir is None:
            log.info("chdir %s " % (chdir))
            os.chdir(chdir)
        pass 
        dir = os.getcwd()
        log.info("running cmd %s from directory %s " % (cmd, dir))
        t0 = time.time()
        pipe = os.popen(cmd)
        ret = pipe.read().strip()
        t1 = time.time()
        rc = pipe.close()
        if rc is None:
            rc = 0 
        rc = os.WEXITSTATUS(rc)
        runtime = t1 - t0

        return_ = self.cnf['return']
        if return_ == 'int': 
            val = self.interpret_as_int(ret)
        elif return_ == 'dict' or return_ == 'json':
            # no simple way to store a string dict or json in sqlite table in a digested form, so leave as a string 
            val = 0
        else:
            raise Exception("return_ %s not handled" % return_ )
            val = None
        pass
        log.info("ret %s val %s rc %s runtime %s return_ %s  " % (ret, val, rc, runtime, return_ )) 
        if val != None:
            kwa = dict(val=val, date=datetime.fromtimestamp(t0).strftime("%Y-%m-%dT%H:%M:%S"))
            if self.version == '0.2':
                kwa.update(runtime=runtime, ret=ret, rc=rc) 
            self.tab.add(**kwa) 
            self.tab.insert()

    def ls(self):
        """
        Dump all entried in the configured table
        """
        sql = "select * from %(tn)s " % self.cnf
        for d in self.tab(sql):
            print d


    def cnf_(self):
        """
        Dump the config for the section
        """
        return str(self.cnf)

    def rep(self):
        """
        Use sqlite3 binary to present the last few entries in the configured table
        """
        hdr = str(self.cnf)
        bdy = os.popen("echo \"select * from %(tn)s order by date desc limit 24 ;\" | sqlite3 %(dbpath)s " % self.cnf).read() 
        return "\n".join(["",hdr,"",bdy])

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
        ret = last.pop('ret',None)

        if ret and self.cnf.get('return',None) == 'dict':
            dret = self.interpret_as_dict(ret)
            log.info("interpreted %s into %s " % ( ret, str(dret) ))
        else:
            dret = None
       
        for k,v in last.items():
            ctx[k] = v

        if dret:
            for k,v in dret.items():
                ctx[k] = self.float_or_asis(v)    

        for k,v in self.cnf.items():
            ctx[k] = self.float_or_asis(v)    

        log.debug(pformat(ctx)) 

        strip_ = lambda _:_.strip().lstrip()
        constraints = strip_(self.cnf['constraints'])
        assert constraints[0] == '(' and constraints[-1] == ')', ("unexpected constraints", constraints )
        lbls = filter(len,map(strip_,constraints[1:-1].split(",")))
        evls = eval(constraints, ctx, {} )
        del ctx['__builtins__']             # rm messy side effect of the eval  
        assert len(lbls) == len(evls)
        edict = dict(zip(lbls,evls))
        return edict, ctx

    def present_edict(self, edict, ctx, constraint=None, summary=False):
        """
        :param edict:
        :param constraint: None, True, False where None corresponds to True OR False

        Presesent the edict 
        """
        all = edict.keys()
        if constraint is None:
            keys = all
        else:
            keys = filter(lambda _:edict[_] == constraint, all) 

        true_ = filter(lambda k:edict[k] == True, all)
        false_ = filter(lambda k:edict[k] == False, all)

        # summarize context, plucking just qtys that are used in the constraints 
        sctx = {}
        for k in all:
           for e in k.lstrip().strip().split():
               if e in ctx:
                   sctx[e] = ctx[e]

        log.debug("sctx:\n%s" % pformat(sctx))

        if summary: 
            fmt = "   %s : %s "
        else:
            fmt = "   %-40s : %s "

        psmy = fmt % ("summary", "all:%s True:%s False:%s " % (len(all),len(true_),len(false_)))
        pctx = fmt % ("context", repr(sctx))
        items = [ fmt % (k,edict[k]) for k in sorted(keys, key=lambda _:edict[_]) ]

        if summary:
            return "".join(items)   # CAUTION this must return a zero length string when all is peachy
        else:
            return "\n".join( [""] + items + ["",psmy,pctx,""] )


    def msg(self):
        """
        Report preparation
        """
        last = self.tab.iterdict("select * from %(tn)s order by date desc limit 1" % self.cnf).next()
        edict, ctx = self.constrain_(last)
        log.debug("ctx:\n %s " % pformat(ctx))
        log.debug("edict:\n %s " % pformat(edict))

        subj = "%(node)s : valmon.py -s %(sect)s : %(date)s : " % dict(last, sect=self.cnf['sect'], node=platform.node() ) 
        smry = self.present_edict(edict, ctx, constraint=False, summary=True )
        subj += smry 
        pass
        if len(smry) == 0:
            msg = ""
        else:
            msg = "\n".join([subj, self.present_edict(edict, ctx, constraint=None, summary=False), self.rep()]) 
        pass
        log.info("subj: %s " % subj )
        log.info("msg : %s " % msg )
        return msg  

    def mon(self):
        """
        Examine last entry in the configured DB table, and compares value to configured constraints
        When excursions are seen send notification email, to configured email addresses
        """
        msg = self.msg()
        if len(msg) > 0:
            subj = msg.split("\n")[0]
            log.warn("sending notification as: %s " % subj )
            notify(self.cnf['email'], msg )
        else:
            log.info("no notification" )

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
            elif arg == 'msg':  # same as mon but doesnt send notification emails, for report development
                self.msg()
            elif arg == 'rep':
                print self.rep()
            elif arg == 'cnf':
                print self.cnf_()
            else:
                log.warn("unhandled arg %s " % arg ) 


class Cnf(dict):
    def __init__(self, cnfpath="~/.env.cnf" ):
        """
        Read `sect` section of config file into this dict

        :param cnfpath: config file path
        """
        dict.__init__(self)
        self['sect'] = None
        cpr = ConfigParser()
        cpr.read(os.path.expanduser(cnfpath))
        pass
        self.cpr = cpr
        self.cnfpath = cnfpath
        self.sections = cpr.sections()

    def has_sect(self, sect):
        return sect in self.sections 

    def read(self, sect):
        """
        :param sect: section name in 
        """
        if not self.has_sect(sect):
            log.info("no section '%s' amongst the cnf '%s' sections %s " % (sect, self.cnfpath,str(self.sections) ))  
            assert 0
        for k,v in self.cpr.items(sect):# spell it out for py2.3
            self[k] = v 
        self['sect'] = sect
        self['sections'] = self.sections
        if not 'email' in self:
           mailto = os.environ.get('MAILTO',None)
           if mailto:
               log.info("no email section configured, but MAILTO envvar is defined, so use that: %s" % mailto ) 
               self['email'] = mailto
           else:
               log.warn("no email section configures and no MAILTO envvar, NOTIFICATION WILL FAIL")

    def __str__(self):
        node = platform.node()
        user = pwd.getpwuid(os.getuid())[0]
        cmt = "%% %s %s@%s " % ( self.cnfpath, user, node )   
        hdr = "[%s]" % self['sect'] 
        skip = "sect sections".split()
        bdy = "\n".join( ["%s = %s " % (k, self[k]) for k in filter(lambda k:k not in skip,self.keys()) ] )
        return "\n".join([cmt, hdr, bdy])

    def __repr__(self):
        return "%s %s %s %s " % ( self.__class__.__name__, self.cnfpath, str(self.sections), repr(dict(self)) )
   

def parse_args(doc):
    """
    Return config dict and commandline arguments 

    :param doc:
    :return: cnf, args  
    """
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-l", "--loglevel",   default="INFO", help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-c", "--cnfpath",   default="~/.env.cnf", help="path to config file Default %default"  )
    op.add_option("-s", "--sect",      default=None , help="section of config file... Default %default"  )
    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )

    if opts.logpath:  # logs to file as well as console, needs py2.4 + (?)
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        try: 
            logging.basicConfig(format=opts.logformat,level=level)
        except TypeError:
            hdlr = logging.StreamHandler()              # py2.3 has unusable basicConfig that takes no arguments
            formatter = logging.Formatter(opts.logformat)
            hdlr.setFormatter(formatter)
            log.addHandler(hdlr)
            log.setLevel(level)
        pass
    pass

    log.info(" ".join(sys.argv))
    #logging.getLogger().setLevel(loglevel)
    return opts, args

def main():
    opts, args = parse_args(__doc__)
    cnf = Cnf(opts.cnfpath)
    log.debug("reading config from sect %s of %s :\n%s " % (opts.sect, opts.cnfpath, cnf))  

    if opts.sect is None:
        msg = "must specify one of the sections in %s  \n " % opts.cnfpath + "\n".join(map(lambda _:"   %s" % _, cnf.sections))
        log.fatal(msg)
        return 
        
    if cnf.has_sect(opts.sect):
        cnf.read(opts.sect) 
        ValueMon(cnf)(args)
    else: 
        log.info("no such section %s in conf : %s " % (opts.sect, repr(cnf)) )


if __name__ == '__main__':
    main()


