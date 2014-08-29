#!/usr/bin/env python
"""

See :env:`/sysadmin/cms02` for example of use of this tool


::

    grep Jun/2014 access_log > Jun_2014_access_log 

    delta:cms02 blyth$ LOG=Jun_2014_access_log DAY=19/Jun/2014 apachehits.py
    grep 19/Jun/2014:00 Jun_2014_access_log | wc -l  421
    grep 19/Jun/2014:01 Jun_2014_access_log | wc -l  383
    grep 19/Jun/2014:02 Jun_2014_access_log | wc -l  422
    grep 19/Jun/2014:03 Jun_2014_access_log | wc -l  355
    grep 19/Jun/2014:04 Jun_2014_access_log | wc -l  379
    grep 19/Jun/2014:05 Jun_2014_access_log | wc -l  2794
    grep 19/Jun/2014:06 Jun_2014_access_log | wc -l  4
    grep 19/Jun/2014:07 Jun_2014_access_log | wc -l  0
    grep 19/Jun/2014:08 Jun_2014_access_log | wc -l  0
    grep 19/Jun/2014:09 Jun_2014_access_log | wc -l  0
    grep 19/Jun/2014:10 Jun_2014_access_log | wc -l  0
    grep 19/Jun/2014:11 Jun_2014_access_log | wc -l  149
    grep 19/Jun/2014:12 Jun_2014_access_log | wc -l  129
    grep 19/Jun/2014:13 Jun_2014_access_log | wc -l  137
    grep 19/Jun/2014:14 Jun_2014_access_log | wc -l  200
    ...


183.60.119.35



"""
from datetime import datetime
import os, sys

def main():
    log = os.environ.get('LOG', 'access_log')
    day = os.environ.get('DAY',datetime.now().strftime("%d/%b/%Y"))  # 20/Jun/2014
    ip = os.environ.get('IP',None)  

    if not ip is None:
        ip_ = "grep ^%(ip)s"  
    else:
        ip_ = None
    pass

    cmds = filter(None,
                   ["cat %(log)s", 
                    ip_,
                    "grep %(day)s:%(hour)0.2d",
                    "wc -l"]) 

    for hour in range(24):
        cmd = "|".join(cmds) % locals()
        n = os.popen(cmd).read().strip()
        print cmd, n 
    pass

if __name__ == '__main__':
    main()

