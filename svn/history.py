#!/usr/bin/env python
"""
Prepare a summary of commit messages for the invoking USER 
as a reminder of activities when writing reports

Usage::
   cd NuWa-trunk/dybgaudi
   python ~/e/svn/history.py > history.txt

See svnlog- ``~/env/tools/svnlog.py`` for a more standalone approach to doing this.


"""
import os, sys
from datetime import timedelta

sys.path.insert(0, os.path.expandvars("$DYB/installation/trunk/dybinst/scripts") )
from svnlog import SVNLog

if __name__ == '__main__':
    user = os.environ.get('USER')
    base = "/dybgaudi/trunk"
    slog = SVNLog(".", "12234:8000" , dict(limit=10000), maxage=timedelta(days=400) ) 
    lastmonth = ""
    for le in filter(lambda _:_.author == user, slog):
       paths = map(lambda _:_.path, le ) 
       location = os.path.commonprefix(paths)   ## string based ... so partial paths happen
       date = le.date[0:10]
       if location.startswith(base):
           location = location[len(base):]
       month = date[5:7]
       if month != lastmonth:
           print "\n\n"
           lastmonth = month
       print "%s %s\n           %s" % ( date, location, le.msg.split("\n")[0] )
       #for path in paths:
       #    print path




