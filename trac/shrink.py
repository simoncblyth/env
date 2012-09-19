#!/usr/bin/env python
"""
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


def builds_(where="cast(rev as int) < 10000"):
    sql = "select id from bitten_build where %(where)s " % locals()
    return sql 

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)


    shk = Shrink(action="select count(*)", bid=10000)
    sql = repr(shk)
    path = "/tmp/env/shrink/count.sql" 
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    log.info("writing to %s " % path )
    with open(path,"w") as fp:
        fp.write(sql)
    #print sql  


