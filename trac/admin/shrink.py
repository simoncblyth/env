#!/usr/bin/env python
"""
Usage::

   ~/e/trac/admin/shrink.py

Extract trac.db from tarball for testing shrinkage at pysqlite level::

    tmp=/tmp/env/blyth/shrink && mkdir -p $tmp && cd $tmp && time tar zxvf /var/scm/backup/dayabay/tracs/dybsvn/2012/09/18/120001/dybsvn.tar.gz dybsvn/db/trac.db

SQLite Versions::

   ============================   ========================    ==========================================================
     Node                           SQLite version              notes
   ============================   ========================    ==========================================================
     C2 /usr/bin/sqlite3            3.3.6                     
     C2 source python pysqlite      3.3.16                     source python 2.5.6, different SQLite version cf sys
     G /opt/local/bin/sqlite3       3.7.11
   ============================   ========================    ==========================================================

TODO
~~~~~

#. group by query to check assumptions wrt build id and revision numbers

   * could use before and after of some such query as zeroth order validity check 

#. command line argument parsing 
#. timings, including line by line
#. recording what was done, including DB file size
#. vacuuming 
#. integrity verification at pysqlite level

    * how to do this ?
    * maybe grouped digests of table content to verify only the expected change gets done

#. graft into trac admin


CHECKED
~~~~~~~

#. SQLite FK support, cascading deletes (from SQLite version 3.6.19)
   
   * http://www.sqlite.org/foreignkeys.html
   * not widely deployed on my nodes and in any case requires schema changes to support this stuff, so a no no

"""
from __future__ import with_statement
import os, logging, sys, sqlite3
from env.sqlite.db import DB
log = logging.getLogger(__name__)

class Shrink(dict):
    """
    The build id are monotonic with rev (or nearly so) : can just kill up to a certain manually determined build id

    TODO:

    * devise group by queries to verify the above statement
      ( note that the rev is text, to support git and others with digest revisions )
    * check for any cascade deleting docs wrt SQLite    

    Usage::

         shk = Shrink(action="select count(*)", bid=10000)
         sql = repr(shk)
         shk = Shrink(action="DELETE", bid=10000)
         sql = repr(shk)

    """
    tmpl = r"""

%(action)s FROM bitten_log_message WHERE log < (SELECT max(id) FROM bitten_log WHERE build < %(bid)s ) ;
%(action)s FROM bitten_report_item WHERE report < (SELECT max(id) FROM bitten_report WHERE build < %(bid)s ) ;

%(action)s FROM bitten_build  WHERE id < %(bid)s  ;

%(action)s FROM bitten_error  WHERE build < %(bid)s ;
%(action)s FROM bitten_step   WHERE build < %(bid)s ; 
%(action)s FROM bitten_slave  WHERE build < %(bid)s ;
%(action)s FROM bitten_log     WHERE build < %(bid)s ; 
%(action)s FROM bitten_report  WHERE build < %(bid)s ;

"""
    __repr__ = lambda self:self.tmpl % self

    def write( self, path ):
        """
        :param path: to write the string repr to  
        """
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        log.info("writing to %s " % path )
        with open(path,"w") as fp:
            fp.write(repr(self))



def builds_(where="cast(rev as int) < 10000"):
    sql = "select id from bitten_build where %(where)s " % locals()
    return sql 

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    base="/tmp/env/blyth/shrink"
    db = DB(path="%s/dybsvn/db/trac.db" % base, skip="bitten")
    print db.tables

    bid=10000

    count_path = "/tmp/env/shrink/count.sql" 
    count = Shrink(action="select count(*)", bid=bid)
    count.write(count_path )

    delete_path = "/tmp/env/shrink/delete.sql" 
    delete = Shrink(action="delete", bid=bid)
    delete.write(delete_path )

    #db.line_by_line( count_path )
    #db.arbitary_( count_path )

    db.arbitary_( delete_path )

