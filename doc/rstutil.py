#!/usr/bin/env python
    
import os, commands, logging
log = logging.getLogger(__name__)

   
def rst2html_open(rst, name, dir_="/tmp"):
    assert type(rst) is unicode, (rst, type(rst))

    rstp = os.path.join(dir_, "%s.rst" % name)
    html = os.path.join(dir_, "%s.html" % name)

    open(rstp, "w").write(rst.encode("utf-8") )

    cmds = [ "rst2html.py %(rstp)s %(html)s", "open %(html)s"]

    for cmd_ in cmds:
        cmd = cmd_ % locals()
        print cmd 
        rc, out = commands.getstatusoutput(cmd)
        if rc != 0:
            log.fatal(" non-zero rc %s : %s " % (rc, out )) 
        pass
        #assert rc == 0, (rc, out) 
    pass


