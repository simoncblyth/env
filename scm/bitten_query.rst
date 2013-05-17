Bitten Querying
================

::

   select build, stopped-started, datetime(started, 'unixepoch'), description, status from bitten_step where name='test-fmcp11a' and build > 20000 ;



::

	sqlite> select * from bitten_report where step='test-fmcp11a' limit 10  ; 
	id          build       step          category    generator                                     
	----------  ----------  ------------  ----------  ----------------------------------------------
	27188       7593        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27197       7600        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27205       7598        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27214       7605        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27226       7608        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27232       7606        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27243       7599        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27248       7603        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27254       7587        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	27263       7611        test-fmcp11a  test        http://bitten.cmlenz.net/tools/python#unittest
	sqlite> 
	sqlite> 
	sqlite> .schema bitten_report 
	CREATE TABLE bitten_report (
	    id integer PRIMARY KEY,
	    build integer,
	    step text,
	    category text,
	    generator text
	);
	CREATE INDEX bitten_report_build_step_category_idx ON bitten_report (build,step,category);
	sqlite> 



::

	sqlite> select id, rev, rev_time, datetime(rev_time,'unixepoch') as rev_time_, started, datetime(started,'unixepoch') as started_ from bitten_build where slave='daya0004.rcf.bnl.gov' and config='dybinst' limit 10 ; 
	id          rev         rev_time    rev_time_            started     started_           
	----------  ----------  ----------  -------------------  ----------  -------------------
	19302       18968       1355253030  2012-12-11 19:10:30  1355254886  2012-12-11 19:41:26
	19314       18978       1355287306  2012-12-12 04:41:46  1355350984  2012-12-12 22:23:04
	19326       18995       1355374164  2012-12-13 04:49:24  1355376252  2012-12-13 05:24:12
	19338       18998       1355384445  2012-12-13 07:40:45  1355389416  2012-12-13 09:03:36
	19350       19007       1355403278  2012-12-13 12:54:38  1355434169  2012-12-13 21:29:29
	19362       19055       1355799195  2012-12-18 02:53:15  1355830227  2012-12-18 11:30:27
	19374       19056       1355808998  2012-12-18 05:36:38  1355811183  2012-12-18 06:13:03
	19386       19073       1355884405  2012-12-19 02:33:25  1355886502  2012-12-19 03:08:22
	19398       19090       1355979495  2012-12-20 04:58:15  1355981592  2012-12-20 05:33:12
	19410       19093       1355984615  2012-12-20 06:23:35  1355995670  2012-12-20 09:27:50
	sqlite> 

