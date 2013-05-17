#!/usr/bin/env python
"""
`altquery.py` for `trac.db` querying
======================================

.. warning:: **never** use this on a live `trac.db`, always extract one from a backup tarball and query that 

Usage:

#. define envvar `DBPATH` to the path to the extracted `trac.db`  (a 7 GB file)
#. if no recent extraction, use `altbackup.py extract_tracdb`

Examples::

       altquery.py last_by_slave
       altquery.py step_times_byrev -s ALL

python version
----------------

Some commands use `group_concat` requiring a recent sqlite. 
The one that comes with py27 is suitable.

sub-commands
-------------

`last_by_slave`
~~~~~~~~~~~~~~~~

::

	[blyth@cms01 scm]$ ./altquery.py last_by_slave
	2013-05-17 19:02:41,419 __main__ INFO     ./altquery.py last_by_slave
	 slave                 id      rev         lastrevtime             laststart             
	 belle7.nuu.edu.tw     20652   20479       2013-05-08 00:23:01     2013-05-08 03:40:09   
	 daya0001.rcf.bnl.gov  20654   20479       2013-05-08 00:23:01     2013-05-08 08:43:41   
	 farm4.dyb.local       20719   20536       2013-05-15 20:23:03     2013-05-15 21:36:59   
	 lxslc507.ihep.ac.cn   20720   20536       2013-05-15 20:23:03     2013-05-15 23:42:34   
	 pdyb-03               20717   20536       2013-05-15 20:23:03     2013-05-16 00:51:11   
	 farm2.dyb.local       20715   20536       2013-05-15 20:23:03     2013-05-16 01:20:45   
	 lxslc504.ihep.ac.cn   20716   20536       2013-05-15 20:23:03     2013-05-16 01:46:54   
	 pdyb-02               20721   20536       2013-05-15 20:23:03     2013-05-16 02:27:27   
	 daya0004.rcf.bnl.gov  20718   20536       2013-05-15 20:23:03     2013-05-16 02:49:22   
	[blyth@cms01 scm]$ 


`step_times_byrev`
~~~~~~~~~~~~~~~~~~~~

::

	[blyth@cms01 scm]$ ./altquery.py step_times_byrev -s test-fmcp11a
	2013-05-17 18:46:36,297 __main__ INFO     ./altquery.py step_times_byrev -s test-fmcp11a
	bitten_build_step_times_byrev   config dybinst  step test-fmcp11a   modifier -30 days   dbpath /data/env/tmp/tracs/dybsvn/2013/05/16/104702/dybsvn/db/trac.db 

	    rev       rev_time                              belle7.nuu.edu.tw     daya0001.rcf.bnl.gov  daya0004.rcf.bnl.gov  farm2.dyb.local       lxslc504.ihep.ac.cn   pdyb-03               
	    20370     2013-04-18 09:19:02                   6352                  -                     -                     5823                  5320                  6929                  
	    20386     2013-04-20 06:39:31                   1592                  -                     -                     1472                  -                     1705                  
	    20394     2013-04-22 21:52:00                   1594                  -                     -                     -                     -                     1697                  
	    20400     2013-04-23 15:31:44                   9817                  -                     -                     -                     -                     10431                 
	    20404     2013-04-24 15:46:54                   1633                  -                     -                     1471                  -                     1911                  
	    20405     2013-04-24 20:51:16                   1609                  -                     -                     1470                  -                     1864                  
	    20411     2013-04-25 19:44:08                   1582                  -                     -                     1476                  -                     1656                  
	    20418     2013-04-26 06:04:23                   1602                  1049                  -                     1483                  -                     1775                  
	    20446     2013-04-30 21:53:33                   1599                  1054                  -                     1478                  -                     1756                  
	    20453     2013-05-02 11:43:14                   1600                  -                     1047                  1488                  1334                  1723                  
	    20471     2013-05-07 08:36:53                   1599                  -                     1052                  1465                  -                     1669                  
	    20477     2013-05-07 22:49:37                   1589                  -                     1043                  1463                  -                     1706                  
	    20479     2013-05-08 00:23:01                   1599                  1045                  -                     1464                  1449                  1869                  
	    20520     2013-05-14 05:56:46                   -                     -                     4268                  5804                  5489                  7073                  
	    20530     2013-05-15 02:40:52                   -                     -                     4230                  5803                  -                     -                     
	    20531     2013-05-15 05:42:24                   -                     -                     1047                  1486                  1420                  2356                  
	    20534     2013-05-15 16:45:42                   -                     -                     1048                  1509                  1395                  1911                  
	    20535     2013-05-15 18:58:10                   -                     -                     1047                  -                     -                     -                     
	    20536     2013-05-15 20:23:03                   -                     -                     1052                  1466                  -                     -                     

"""
import os, logging, sys
log = logging.getLogger(__name__)
from db import DB
os.environ.setdefault('DBPATH','/data/env/tmp/tracs/dybsvn/2013/05/16/104702/dybsvn/db/trac.db')

dt_ = lambda field,label:" datetime(%(field)s,'unixepoch') as %(label)s" % locals()
recent_ = lambda field, modifier:" datetime(%(field)s,'unixepoch') > datetime('now','%(modifier)s')" % locals()


class TracDB(DB):
    def configs(self):
        return map(lambda _:_['config'], self("select distinct(config) as config from bitten_build where datetime(started,'unixepoch') > datetime('now', '-30 days') " % locals()))
    def slaves(self, config='dybinst'):
        return map(lambda _:_['slave'], self("select distinct(slave) as slave from bitten_build where datetime(started,'unixepoch') > datetime('now', '-30 days') and config='%(config)s' " % locals()))
    def steps(self, config='dybinst'):
        return map(lambda _:_['name'], self("select distinct(name) as name from bitten_step where datetime(started,'unixepoch') > datetime('now', '-30 days') " % locals()))

def bitten_build_last_by_slave_(**kwa):
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

def bitten_build_last_by_slave(**kwa):
    sql, keys, fmt  = bitten_build_last_by_slave_(**kwa) 
    print fmt % dict((k,k) for k in keys)
    db = TracDB()
    for d in db(sql):
        print fmt % d

def bitten_build_step_times_(config='dybinst',step_name='test-fmcp11a', slave='pdyb-03', modifier='-30 days'):
    """
    :: 

	sqlite> select bb.*, bs.*  from bitten_build bb join bitten_step bs on bb.id = bs.build where datetime(bb.started,'unixepoch') > datetime('now','-30 days') and bb.config = 'dybinst' and bs.name = 'test-fmcp11a' limit 10  ;
	id          config      rev         rev_time    platform    slave              started     stopped     status      build       name          description  status      started     stopped   
	----------  ----------  ----------  ----------  ----------  -----------------  ----------  ----------  ----------  ----------  ------------  -----------  ----------  ----------  ----------
	20532       dybinst     20370       1366276742  15          belle7.nuu.edu.tw  1366278060  1366304741  S           20532       test-fmcp11a               S           1366290953  1366297305
	20535       dybinst     20370       1366276742  32          farm2.dyb.local    1366281842  1366306796  S           20535       test-fmcp11a               S           1366294416  1366300239
	20536       dybinst     20370       1366276742  31          lxslc504.ihep.ac.  1366342592  1366369126  S           20536       test-fmcp11a               S           1366344747  1366350067
	20537       dybinst     20370       1366276742  35          pdyb-03            1366312553  1366335729  S           20537       test-fmcp11a               S           1366317082  1366324011
	20542       dybinst     20386       1366439971  15          belle7.nuu.edu.tw  1366441289  1366458088  S           20542       test-fmcp11a               S           1366449221  1366450813
	20545       dybinst     20386       1366439971  32          farm2.dyb.local    1366441424  1366455995  S           20545       test-fmcp11a               S           1366448054  1366449526
	20547       dybinst     20386       1366439971  35          pdyb-03            1366470686  1366487784  S           20547       test-fmcp11a               S           1366474664  1366476369
	20552       dybinst     20394       1366667520  15          belle7.nuu.edu.tw  1366668971  1366688722  S           20552       test-fmcp11a               S           1366679933  1366681527
	20557       dybinst     20394       1366667520  35          pdyb-03            1366698475  1366715888  F           20557       test-fmcp11a               S           1366702685  1366704382
	20562       dybinst     20400       1366731104  15          belle7.nuu.edu.tw  1366885211  1366908983  S           20562       test-fmcp11a               S           1366892008  1366901825
	sqlite> 

    """
    cols = []
    # from bitten_build 
    cols += ["build.id bld","build.rev rev","build.status bstat", "build.slave slave"  ]
    cols += ["( build.stopped - build.started ) as build_secs"]
    cols += [dt_("build.rev_time", "rev_time")]
    cols += [dt_("build.started","build_started")]
    # from bitten_step
    cols += ["( step.stopped - step.started ) as step_secs"]
    cols += ["step.name step_name", "step.status step_status"]
    cols += [dt_("step.started","step_started")]

    cols = ",".join(cols)

    from_ = "bitten_build build join bitten_step step on build.id = step.build"   
    where=" build.config='%(config)s' and step.name='%(step_name)s' " % locals()
    if slave:
        where += " and build.slave='%(slave)s' " % locals()
    where += " and " + recent_("build.rev_time",modifier)

    sql = "select %(cols)s from %(from_)s where %(where)s limit 300" % locals()
    return sql 


def bitten_build_step_times(config='dybinst',step_name='test-fmcp11a', slave='pdyb-03', modifier='-30 days'):
    fmt = "  %(rev)-6s  %(slave)-30s  %(rev_time)s   %(build_secs)-5s    %(step_name)s        %(step_secs)s  "
    db = TracDB()
    print db.slaves()
    for slave in db.slaves():
        sql = bitten_build_step_times_(slave=slave)
        for d in db(sql):
            print fmt % d


def bitten_build_step_times_byrev_(**kwa):
    """
    select rev, count(*) as nbuild from bitten_build build join bitten_step step on build.id = step.build where datetime(build.started,'unixepoch') > datetime('now','-30 days') and build.config = 'dybinst' and step.name = 'test-fmcp11a' group by rev ;  
    """
    pass
    cols = []
    cols += ["rev", dt_("min(rev_time)","rev_time"), "count(*) as nbuild"]
    cols += ["\"{\"||group_concat(\"'\"||build.slave||\"'\"||\":\"||(step.stopped - step.started))||\"}\" as slavesecs"]
    cols = ",".join(cols)
    from_ = "bitten_build build join bitten_step step on build.id = step.build"   
    where=" build.config='%(config)s' and step.name='%(step_name)s' " % kwa
    where += " and " + recent_("build.rev_time",kwa['modifier'])
    kwa.update(cols=cols, where=where, from_=from_)
    sql = "select %(cols)s from %(from_)s where %(where)s group by rev " % kwa
    return sql 

def bitten_build_step_times_byrev(**kwa):
    """
    """
    log.debug("needs sqlite3 version >= 3.5.4 for group_conat , py27 such as nuwa python has this " )

    db = TracDB()
    slaves = sorted(db.slaves())
    kwa['dbpath'] = db.path

    def present_slavesecs(**kwa):
        sql = bitten_build_step_times_byrev_(**kwa)
        print "bitten_build_step_times_byrev   config %(config)s  step %(step_name)s   modifier %(modifier)s   dbpath %(dbpath)s " % kwa
        print 
        sfmt =  " %-20s " 
        fmt  = "    %(rev)-7s   %(rev_time)-30s       %(slavesecs)s "  
        print fmt % dict(rev="rev",rev_time="rev_time", slavesecs="".join(map(lambda _:sfmt % _, slaves )))
        for d in db(sql):
            ss = eval(d['slavesecs'])
            d['slavesecs'] = "".join( map(lambda _:sfmt % ss.get(_,"-"), slaves ))        
            print fmt % d
    pass

    if kwa['step_name'] == 'ALL':
        steps = sorted(db.steps())
        for step in steps:
            kwa['step_name'] = step
            present_slavesecs(**kwa)
    else: 
        present_slavesecs(**kwa)



def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-l", "--loglevel", default="info")
    op.add_option("-c", "--config", default="dybinst" )
    op.add_option("-s", "--step", dest="step_name", default="test-fmcp11a" )
    op.add_option("-d", "--days", default="30" , help="Restrict revision range to start this number of days before. Default %default " )
    opts, args = op.parse_args()
    level=getattr(logging,opts.loglevel.upper()) 
    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))
    return opts, args

def main():    
    opts, args = parse_args_(__doc__)
    kwa = vars(opts)
    kwa['modifier'] = "-%(days)s days" % kwa 

    for arg in args:
        if arg == "last_by_slave":
            bitten_build_last_by_slave(**kwa)
        elif arg == "step_times":
            bitten_build_step_times(**kwa)
        elif arg == "step_times_byrev":
            bitten_build_step_times_byrev(**kwa)
        else:
            log.warn("unhandled arg %s " % arg )


if __name__ == '__main__':
    main()

