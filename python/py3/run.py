#!/usr/bin/env python
"""
"""
from __future__ import with_statement
import os, hashlib, logging, re, sys, subprocess
from functools import partial

try:
    import commands
except ImportError:
    commands = None
pass

log = logging.getLogger(__name__)

def run(cmd):
    if not commands is None:
        rc, out = commands.getstatusoutput(cmd)
    else:
        #cpr = subprocess.run(cmd.split(" "), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cpr = subprocess.run(cmd.split(" "), check=True, text=True, capture_output=True)
        out = cpr.stdout
        rc = cpr.returncode 
        #print(str(out))
    pass
    if rc != 0:
        log.fatal("non-zero RC from cmd : %s " % cmd ) 
    pass    
    assert rc == 0,  rc  
    log.debug("cmd:[%s] out:%d " % (cmd, len(out)) ) 
    return out 


if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, "INFO"))

    cmd1 = "git status --porcelain"
    cmd2 = cmd1 + " notes/cn/China_Payroll_New_Rules_for_Individual_Income_Tax.pdf"
    print("##############")
    out1 = run(cmd1)
    print(out1)
    print("##############")
    out2 = run(cmd2)
    print(out2)
    print("##############")

    


