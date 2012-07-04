#!/usr/bin/env python
"""
"""
from fabric.api import run
from env.tools.libfab import fabloop

def check(host):
    return run("uname -a")	

if __name__ == '__main__':
    ret = fabloop('C C2 H N N1', check)
    print ret 

