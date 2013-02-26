Altbackup
===========

Failed to reproduce the below issue
------------------------------------


::

    === altbackup_notify: FAILURE Mon Feb 25 23:00:03 CST 2013 /home/blyth/cronlog/altbackup.log cms01.phys.ntu.edu.tw 

    === altbackup_main: log truncate Mon Feb 25 23:00:01 CST 2013
    2013-02-25 23:00:02,548 __main__ INFO     <Values at 0xb7f2976c: {'wanted': 'dybsvn svnsetup', 'target': None, 'targetnode': 'C', 'loglevel': 'INFO', 'logpath': '/home/blyth/cronlog/altbackup.log', 'ext': '.tar.gz', 'echo': False, 'source': None, 'logformat': '%(asctime)s %(name)s %(levelname)-8s %(message)s', 'keep': 3}>
    2013-02-25 23:00:02,548 __main__ INFO     source     : /home/scm/backup/dayabay 
    2013-02-25 23:00:02,549 __main__ INFO     target     : /data/var/scm/alt.backup/dayabay 
    2013-02-25 23:00:02,549 __main__ INFO     alt_check /data/var/scm/alt.backup/dayabay ['dybsvn', 'svnsetup'] 
    2013-02-25 23:00:02,549 __main__ INFO     looking for ['dybsvn'] source tarballs beneath /data/var/scm/alt.backup/dayabay from 2013/02/25 
    2013-02-25 23:00:02,679 __main__ WARNING  SKIPPING AS no dna for path /data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/25/104702/dybsvn-19839.tar.gz 
    2013-02-25 23:00:03,503 __main__ INFO     found 0 matching tarballs


::

	[blyth@cms01 cronlog]$ cat altbackup_.log
	=== altbackup_main: /home/blyth/env/scm/altbackup.py -o /home/blyth/cronlog/altbackup.log dump check_target
	Traceback (most recent call last):
	  File "/home/blyth/env/scm/altbackup.py", line 368, in <module>
	    alt_check( target, cfg )
	  File "/home/blyth/env/scm/altbackup.py", line 317, in alt_check
	    assert npaths == expect[want], "expecting %s paths for %s BUT got %s  " % (expect[want], want, relpaths ) 
	AssertionError: expecting 2 paths for dybsvn BUT got []  
	=== altbackup_main: ERROR RC 1
	=== altbackup_notify: FAILURE Mon Feb 25 23:00:03 CST 2013 /home/blyth/cronlog/altbackup.log cms01.phys.ntu.edu.tw : sending notification MAILTO blyth@hep1.phys.ntu.edu.tw
	[blyth@cms01 cronlog]$ 



manual target check
---------------------

::

	[blyth@cms01 dayabay]$ find /data/var/scm/alt.backup/dayabay
	/data/var/scm/alt.backup/dayabay
	/data/var/scm/alt.backup/dayabay/svn
	/data/var/scm/alt.backup/dayabay/svn/dybsvn
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/26
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/26/104701
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/26/104701/dybsvn-19844.tar.gz
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/26/104701/dybsvn-19844.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/25
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/25/104702
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/25/104702/dybsvn-19839.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/25/104702/dybsvn-19839.tar.gz
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/24
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/24/104702
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/24/104702/dybsvn-19839.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/svn/dybsvn/2013/02/24/104702/dybsvn-19839.tar.gz
	/data/var/scm/alt.backup/dayabay/tracs
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/26
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/26/104701
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/26/104701/dybsvn.tar.gz
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/26/104701/dybsvn.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/25
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/25/104702
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/25/104702/dybsvn.tar.gz
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/25/104702/dybsvn.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/24
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/24/104702
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/24/104702/dybsvn.tar.gz
	/data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/02/24/104702/dybsvn.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/folders
	/data/var/scm/alt.backup/dayabay/folders/svnsetup
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/26
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/26/104701
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/26/104701/svnsetup.tar.gz
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/26/104701/svnsetup.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/25
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/25/104702
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/25/104702/svnsetup.tar.gz
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/25/104702/svnsetup.tar.gz.dna
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/24
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/24/104702
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/24/104702/svnsetup.tar.gz
	/data/var/scm/alt.backup/dayabay/folders/svnsetup/2013/02/24/104702/svnsetup.tar.gz.dna
	[blyth@cms01 dayabay]$ 

