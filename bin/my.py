#!/usr/bin/env python
"""

.. warn:: Developed in dybsvn, copied here for convenience


Emits mysql client command with authentication parameters appropriate 
for the DBCONF passed as a single argument 

This is a standalone version of the NuWa based mysql.py to allow easy usage 
from anywhere. Requires python, mysql and my.py in the PATH and a ~/.my.cnf
grab with::

   svn export http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/DybPython/scripts/my.py

NB based on ~/.my.cnf config file, get that when moving to new node::

    delta:~ blyth$ scp CN:.my.cnf .


Usage::

   echo select \* from LOCALSEQNO | $(my.py offline_db) 
   echo status | $(my.py tmp_offline_db)
   $(my.py offline_db)    # commandline client

OR more conveniently add some bash functions to .bash_profile with 
names corresponding to the often used sections of your ~/.my.cnf::

    #idb(){ local cnf=$1 ; shift ; eval $(db.py $cnf cli) $* ; }
    idb(){ local cnf=$1 ; shift ; eval $(my.py $cnf) $* ; }

    offline_db(){             idb $FUNCNAME $* ; }
    tmp_offline_db(){         idb $FUNCNAME $* ; }
    tmp_etw_offline_db(){     idb $FUNCNAME $* ; }
    tmp_jpochoa_offline_db(){ idb $FUNCNAME $* ; }
    dayabaydb_lbl_gov(){      idb $FUNCNAME $* ; }
    dybdb1_ihep_ac_cn(){      idb $FUNCNAME $* ; }
    dybdb2_ihep_ac_cn(){      idb $FUNCNAME $* ; }
    ihep_dcs(){               idb $FUNCNAME $* ; }
   
A higher level *almost* equivalent version of this is available as 
the `cli` sub-command of the `db.py` script.

"""
import os, sys
from datetime import datetime
from ConfigParser import ConfigParser

class MyCnf(dict):
    def __init__(self, path = "~/.my.cnf"): 
        prime = dict(today=datetime.now().strftime("%Y%m%d"))
        cfp = ConfigParser(prime)
        paths = cfp.read( [os.path.expandvars(os.path.expanduser(p)) for p in path.split(":")] )
        self.cfp = cfp
        self.path = path
        self.paths  = paths
    def section(self, sect):
        return dict(self.cfp.items(sect))
    sections = property(lambda self:self.cfp.sections()) 
    def dump(self):
        for sect in self.sections:
            cnf = self.section(sect)
            cmd = MySQLCmd(cnf)
            print cnf,cmd.cmd_nopw

class CommandLine(dict):
    cmd      = property( lambda self:self._cmd % self )
    cmd_nopw = property( lambda self:self._cmd % dict(self, password="***") )

class MySQLCmd(CommandLine):
    """
    Return command without doing it
    """
    _cmd = "%(exepath)s --no-defaults -t --host=%(host)s --user=%(user)s --password=%(password)s %(database)s "

    def __init__(self, *args, **kwa):
        CommandLine.__init__(self, *args, **kwa)
        self['exepath'] = os.popen("which mysql").read().rstrip("\n")  

    def __call__(self):
        return self.cmd


if __name__ == '__main__':
    mycnf = MyCnf()
    if len(sys.argv)==1:
        print "need argument specifing section of %s, ie one of \n%s\n" % ( mycnf.path, "\n".join(map(lambda _:"   %s" % _,sorted(mycnf.sections))) )
    else:
        for sect in sys.argv[1:]:   
            cnf = mycnf.section(sect) 
            cmd = MySQLCmd(cnf)
            print cmd()        




