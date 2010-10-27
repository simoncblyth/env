#!/usr/bin/env python
"""
   Parse ~/.my.cnf and dump into separate .cnf for each section 
   with section label changed to "client"
   
   This allows rapid mysqlclient connection to different
   servers/accounts, eg with a bash function like :

my- () 
{ 
    [ ~/.my.cnf -nt ~/.my/client.cnf ] && python $(env-home)/mysql/splitcnf.py;
    [ -z "$1" ] && ls  -l ~/.my/;
    mysql --defaults-file=~/.my/${1:-client}.cnf
}

   that re-splits as needed :
       * after edits to ~/.my.cnf are made
       * if no directory ~/.my exists

   Note it is required that the ~/.my.cnf has a "client" section,
   as it should do in any case for normal mysql usage.

   my- <secname>

   my- offline_db
   my- testdb
   my- recovered_offline_db
   

"""

import os
from ConfigParser import ConfigParser
from StringIO import StringIO

def parse():
    cfp = ConfigParser()
    cfp.read( os.path.expanduser("~/.my.cnf"))
    return cfp

def prep_savd():
    savd = os.path.expanduser("~/.my") 
    if not os.path.exists(savd):
        os.makedirs(savd)
    assert os.path.isdir(savd)
    return savd

def split(): 
    savd = prep_savd()
    secs = parse().sections()
    for sec in secs:
        zfp = parse()            ## no copy ctor it seems 
        sio = StringIO()
        for xec in secs:
            if xec != sec:
                zfp.remove_section( xec )
        zfp.write( sio )

        s = sio.getvalue()
        s = s.replace( "[%s]" % sec , "[%s]" % "client" )
        s = s.replace( "engine" , "#engine" ) ## needed by django, but mysql client dont like 
        spl = os.path.join( savd , "%s.cnf" % sec )
        open(spl,"w").write( s )
        print "writing to %s " % spl




if __name__=='__main__':
    split()
    








