#!/usr/bin/env python
"""
Extract trac.db from tarball and interactively check with::

	[dayabay] /tmp/tt > tar zxf dybsvn.tar.gz dybsvn/db/trac.db
	[dayabay] /tmp/tt > du -h dybsvn/db/trac.db
	5.5G    dybsvn/db/trac.db
	[dayabay] /tmp/tt > sqlite3 dybsvn/db/trac.db
	SQLite version 3.3.3
	Enter ".help" for instructions
	sqlite> .tables
	attachment          bitten_report       node_change         ticket
	auth_cookie         bitten_report_item  permission          ticket_change
	bitten_build        bitten_rule         report              ticket_custom
	bitten_config       bitten_slave        revision            version
	bitten_error        bitten_step         session             wiki
	bitten_log          component           session_attribute
	bitten_log_message  enum                system
	bitten_platform     milestone           tags
	sqlite> 

Use this script to run some queries on the sqlite db::

        ./count.py /tmp/tt/dybsvn/db/trac.db

	bitten_config                  : 5 
	bitten_platform                : 16 
	bitten_rule                    : 16 
	bitten_error                   : 8164 
	bitten_build                   : 13170 
	bitten_report                  : 98520 
	bitten_slave                   : 162784 
	bitten_log                     : 298469 
	bitten_step                    : 300033 
	bitten_report_item             : 18051235     ## more than 18M 
	bitten_log_message             : 39656123     ## almost 40M 

sqlite> .schema bitten_log_message
CREATE TABLE bitten_log_message (
    log integer,
    line integer,
    level text,
    message text,
    UNIQUE (log,line)
);

sqlite> .schema bitten_report_item  
CREATE TABLE bitten_report_item (
    report integer,
    item integer,
    name text,
    value text,
    UNIQUE (report,item,name)
);

sqlite> select max(length(message)) from bitten_log_message ;
3571

sqlite> select sum(length(message)) from bitten_log_message ;
2048409596
## 2 billion chars of log messages


sqlite> select count(*) FROM bitten_log_message WHERE log < (SELECT max(id) FROM bitten_log WHERE build < 10000 ) ;
10478700

sqlite> SELECT max(id) FROM bitten_log WHERE build < 10000 ;
148781

sqlite> select count(*) FROM bitten_log_message ;
39656123
sqlite> 

sqlite> select cast(rev as integer) from bitten_build limit 10 ;
3953
3952
3957

sqlite> select count(*) from bitten_build where cast(rev as int) < 10000 ;
3175

sqlite> select count(distinct(id)) from bitten_build where cast(rev as int) < 10000 ;
3175

sqlite> select count(distinct(id)) from bitten_build where cast(rev as int) > 10000 ;
9995

sqlite> SELECT min(rev+0),max(rev+0) FROM bitten_build  ;
3952|17979


sqlite> select distinct(config) from bitten_build;
detdesc
dybinst
dybdoc
dybdaily
opt.dybinst



"""
from __future__ import with_statement
import logging, sys, sqlite3
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
    with open(path,"w") as fp:
        fp.write(sql)
    print sql  


