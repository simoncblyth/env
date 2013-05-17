#!/usr/bin/env python
"""
"""
import os, logging
log = logging.getLogger(__name__)
from db import DB
os.environ.setdefault('DBPATH','/data/env/tmp/tracs/dybsvn/2013/05/16/104702/dybsvn/db/trac.db')

dt_ = lambda field,label:" datetime(%(field)s,'unixepoch') as %(label)s" % locals()
recent_ = lambda field, modifier:" datetime(%(field)s,'unixepoch') > datetime('now','%(modifier)s')" % locals()

def bitten_build_last_by_slave_():
    """
    Use the trac.db extracted from the backup tarball to determine the laststart times of the last build
    from each of the slaves, query restricts to last 30 days to skip retired slaves::


	slave                 id          rev         lastrevtime           laststarted         
	--------------------  ----------  ----------  --------------------  --------------------
	lxslc507.ihep.ac.cn   20630       20453       2013-05-02 11:43:14   2013-05-02 12:04:01 
	farm4.dyb.local       20629       20453       2013-05-02 11:43:14   2013-05-02 12:07:55 
	lxslc504.ihep.ac.cn   20626       20453       2013-05-02 11:43:14   2013-05-02 13:03:01 
	belle7.nuu.edu.tw     20652       20479       2013-05-08 00:23:01   2013-05-08 03:40:09 
	daya0001.rcf.bnl.gov  20654       20479       2013-05-08 00:23:01   2013-05-08 08:43:41 
	daya0004.rcf.bnl.gov  20658       20479       2013-05-08 00:23:01   2013-05-08 09:20:28 
	pdyb-02               20661       20479       2013-05-08 00:23:01   2013-05-08 09:48:35 
	farm2.dyb.local       20655       20479       2013-05-08 00:23:01   2013-05-08 11:22:42 
	pdyb-03               20657       20479       2013-05-08 00:23:01   2013-05-09 00:13:52 

    """ 
    cols = ",".join(["slave","max(id) id", "max(rev) rev", "max(rev_time) as _lastrevtime", dt_("max(rev_time)", "lastrevtime"), "max(started) as _laststart", dt_("max(started)","laststart") ])
    keys = ("slave","id","rev","_lastrevtime","lastrevtime","_laststart","laststart",)
    fmt = " %(slave)-20s  %(id)-7s %(rev)-7s     %(lastrevtime)-20s    %(laststart)-20s  " 
    where = recent_("rev_time","-30 days")
    sql = "select %(cols)s from bitten_build where %(where)s group by slave having slave != '' order by _laststart " % locals()
    return sql, keys, fmt

def bitten_build_last_by_slave():
    sql, keys, fmt  = bitten_build_last_by_slave_() 
    print fmt % dict((k,k) for k in keys)
    db = DB()
    for d in db(sql):
        print fmt % d

def bitten_build_where(where="slave='daya0004.rcf.bnl.gov' and config='dybinst'"):
    cols = ["id","status", "rev"]
    tfield = "rev_time started stopped".split()
    cols += tfield
    cols += map( lambda field:dt_(field,field+"_"), tfield )
    cols += ["stopped - started as seconds "]
    cols = ",".join(cols)
    #cols = "*"
    where += " and " + recent_("rev_time","-30 days")
    sql = "select %(cols)s from bitten_build where %(where)s limit 100" % locals()


    db = DB()
    db.externally(sql)
    for d in db(sql):
        print d

def main():
    logging.basicConfig(level=logging.INFO)
    pass
    #bitten_build_last_by_slave()
    bitten_build_where()

if __name__ == '__main__':
    main()

