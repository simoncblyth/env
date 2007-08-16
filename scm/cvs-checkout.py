#!/usr/bin/env python

import pexpect
import os
import sys

def checkout( cvsroot , cvspass , cvstag=None ):
    ''' interact with CVS to allow automated check outs '''
    print " ==== cvs-checkout.py cvsroot:%s cvspass:%s cvstag:%s " % ( cvsroot , cvspass , cvstag ) 
    login(cvsroot,cvspass)
    if cvstag==None or cvstag=="head":
        cmdtag = ""
    else:
        cmdtag = "-r %s" % cvstag 
    cmd = "cvs -d %s checkout %s . " % ( cvsroot , cmdtag )
    try:
        print "invoke [%s] " % cmd 
        s = pexpect.spawn(cmd)
        s.logfile = sys.stdout
        s.interact()
    except:
        print 'Something went wrong trying to run %s ' % cmd
        print 'it returned:'
        print s.before
    return 0


def login(cvsroot,cvspass):
    """ login """
    cmd = 'cvs -d %s login ' % cvsroot
    try:
        print "invoke [%s] " % cmd 
        s = pexpect.spawn(cmd)
        s.logfile = sys.stdout
        i = s.expect (['CVS password: '])
        if i==0:
            s.sendline( cvspass ) 
        else:
            print " i is %s " % i
        s = None
    except:
        print 'Something went wrong trying to run %s ' % cmd
        print 'it returned:'
        print s.before
    return 0


if __name__ == "__main__":
    #checkout( os.environ['DYW_CVSROOT_DAYABAY']  ,  os.environ['DYW_PASS'] , sys.argv[1:] )
    checkout( sys.argv[1:] )
    



#
# cvs -d $DYW_CVSROOT_DAYABAY login 
# Logging in to :pserver:dayabay@dayawane.ihep.ac.cn:2401/home/dybcvs/cvsroot
# CVS password: 
# cvs login: authorization failed: server dayawane.ihep.ac.cn rejected access to /home/dybcvs/cvsroot for user dayabay
#