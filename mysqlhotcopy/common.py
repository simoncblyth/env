#!/usr/bin/env python
"""
"""
import os, time, logging
log = logging.getLogger(__name__)

seconds = {}

def timing(func):
    def wrapper(*arg,**kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        global seconds
        seconds[func.func_name] = (t2-t1)
        return res 
    return wrapper


def do(cmd, verbose=False, stderr=True):
    if not stderr:
        cmd = cmd + " 2>/dev/null"
    if verbose:
        print cmd
    log.debug("do %s " % cmd )
    p = os.popen(cmd,'r')
    ret = p.read().strip()
    rc = p.close()
    log.debug("rc:%s len(ret):%s\n[%s]" % ( rc, len(ret), ret ))
    return rc, ret


def scp( spath, tpath , remotenode="C" , sidecar_ext='' ):
    log.info("transfer %(spath)s %(tpath)s %(remotenode)s %(sidecar_ext)s " % locals() )
    tdir = os.path.dirname(tpath) 
    cmds = [
              "ssh %(remotenode)s \"mkdir -p %(tdir)s \" ", 
              "time scp %(spath)s%(sidecar_ext)s %(remotenode)s:%(tpath)s%(sidecar_ext)s ", 
           ]
    for cmd in cmds:
        rc, ret = do(cmd % locals(),verbose=True)


if __name__ == '__main__':
    pass
