#!/usr/bin/env python
"""
linetab.py
============

Creates sqlite DB with single table containing 
the lines from a txt file, the path to which
is given in the only argument.

For example with txtpath /tmp/somelist.txt
the DB is created at /tmp/somelist.txt.db 
containing table "somelist" with fields : idx,line,priority 

Examine the DB with::

   sqlite3 /tmp/somelist.txt.db 
   > .tables
   > select idx,line from somelist order by priority limit 10 ; 

"""
import os, sys, logging
sys.path.insert(0, os.path.expandvars("$HOME"))

log = logging.getLogger(__name__)
from env.db.simtab import Table

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    txtpath = sys.argv[1] 
    def priority_(line):
        return len(line.split("/")) # dummy example 
    pass
    tab = Table.FromLines(txtpath, priority_) 


