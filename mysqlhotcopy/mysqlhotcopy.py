#!/usr/bin/env python
"""
MySQL Hotcopy wrapper 
=======================

#. avoids filling disk by estimating space required for hotcopy, 
   from DB queries and file system free space checking 
#. creates tarballs in dated folders

Intended to be used in system python from sudo, operating from non-pristine 
env will cause errors related to setuptools.
Requires MySQLdb, check that and operating env with::

    sudo python -c "import MySQLdb"

If that gives errors will need to::

    sudo yum install MySQL-python



Pre-requisites for mysqlhotcopy
--------------------------------

Perl module `DBD::mysql`
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

	install_driver(mysql) failed: Can't locate DBD/mysql.pm 

	[root@belle1 ~]# perl -mDBD::mysql -e ''
	Can't locate DBD/mysql.pm in @INC (@INC contains: /usr/lib/perl5/site_perl/5.8.8/i386-linux-thread-multi /usr/lib/perl5/site_perl/5.8.8 /usr/lib/perl5/site_perl /usr/lib/perl5/vendor_perl/5.8.8/i386-linux-thread-multi /usr/lib/perl5/vendor_perl/5.8.8 /usr/lib/perl5/vendor_perl /usr/lib/perl5/5.8.8/i386-linux-thread-multi /usr/lib/perl5/5.8.8 .).
	BEGIN failed--compilation aborted.


Commands
---------

The first argument of the `mysqlhotcopy.py` script specifies the name of a mysql database
to operate upon.  Subsequent arguments specify actions to take. Order is important. 


`hotcopy`
      use *mysqlhotcopy* to copy the mysql datadir for a single database into a dated folder under `backupdir`, 
      with path of form `/var/dbbackup/mysqlhotcopy/belle1.nuu.edu.tw/channelquality_db/20130530_1954`
      the folder and its parent folders are created if necessary 

`coldcopy`
       Manual file system copy **without locking** the DB. Used for investigating crashed DBs, DO NOT USE FOR RELIABLE BACKUPS. TREAT WITH CAUTION

`archive`
      archives the dated folder into a tarball and deletes the dated folder

`examine`
      examines the paths within the tarball and determines their common prefix path and classifies as flattop or otherwise
      `examine` must be run before doing `extract`

`transfer`
      uses *scp* to copy a tarball to a remote node

`extract`
      extracts the content from a tarball into the mysql datadir, **CAUTION THIS IS DANGEROUS** test on non-critical servers first

`ls`
      list the tarballs for a database

`purge`
      purge the tarballs for a database, keeping the configured number


Features of mysqlhotcopy
---------------------------

mysqlhotcopy does low level file copying, making version closeness important  

::

   dybdb1.ihep.ac.cn        5.0.45-community-log MySQL Community Edition (GPL)
   belle7.nuu.edu.tw        5.0.77-log Source distribution
   cms01.phys.ntu.edu.tw    4.1.22-log

  
Usage steps
-----------

Examples of usage::

    cd env/mysqlhotbackup
    ./mysqlhotbackup.py --help

    ./mysqlhotbackup.py tmp_ligs_offline_db  hotcopy archive transfer 

          # 1st argument is DB name, subsequent are the actions to take
          # the **hotcopy** action is the one during which the DB tables are locked


    ./mysqlhotbackup.py -t 20130516_1711 tmp_ligs_offline_db transfer 

          # if need to transfer or archive separately from the hotcopy 
          # then must specify the time tag corresponding to the hotcopy and archive 
          # to be transferred

    ./mysqlhotcopy.py --regex './^\(DqChannelStatus\|DqChannelStatusVld\)/'  tmp_ligs_offline_db hotcopy archive transfer      

          # using regex to only include 2 tables, this regex is tacked on to the mysqhotcopy 
          # database argument and subsequently interpreted as a perl regular expression 
    
    ./mysqlhotcopy.py -l debug --regex "^DqChannelPacked" tmp_offline_db coldcopy 

          # NB when using coldcopy the regex must be a python compatible regexp to select tables to include

    ./mysqlhotcopy.py --regex './^\(DqChannelPacked\|DqChannelPackedVld\)/'  tmp_offline_db hotcopy       

          # have to escape the brackets and pipe symbol to protect from shell interpretation

    ./mysqlhotcopy.py -C --regex './^LOCALSEQNO/' tmp_offline_db hotcopy archive 

          # for quick machinery testing, restrict to just handling a small table and disable interactive confirmations
          # note that hotcopy will delete a pre-existing same minute folder however

    ./mysqlhotcopy.py -l debug --flattop --remotenode S --regex '^(DqChannelStatus|DqChannelStatusVld)\.[^.]*$'  tmp_ligs_offline_db_0  coldcopy 

         # more realistic coldcopy python regex to just handle a single pair of DBI tables, note that the pattern has to match that of 
         # names of the internal mysql files ie `.MYI` `.MYD` `.frm`

    rm -rf /tmp/tmp_offline_db && mysqlhotcopy.py --regex "^LOCALSEQNO"  -l debug --ALLOWEXTRACT -x /tmp -C tmp_offline_db coldcopy archive examine extract  

          # quick full coldcopy chain testing 

    rm -rf /tmp/tmp_offline_db && mysqlhotcopy.py --regex "./^LOCALSEQNO/"  -l debug --ALLOWEXTRACT -x /tmp -C tmp_offline_db hotcopy archive examine extract  

          # quick full hotcopy chain testing 

    mysqlhotcopy.py --regex "^DqChannelPacked"  -l debug --ALLOWEXTRACT --flattop -C --rename tmp_offline_db_ext tmp_offline_db coldcopy archive examine extract  
 
          # after this, can immediately "use tmp_offline_db_ext" and query against the coldcopied tables 


Ownership issue FIXED
~~~~~~~~~~~~~~~~~~~~~~

During development the ownership of `coldcopy` directories was initially not preserved, until r3745::

     rm -rf /tmp/tmp_offline_db && mysqlhotcopy.py --regex "./^LOCALSEQNO/"  -l debug --ALLOWEXTRACT -x /tmp --flattop -C tmp_offline_db hotcopy archive examine extract  && ll tmp_offline_db/ &&  [ $(id -u mysql) -eq $(stat -c %u /tmp) ] && echo OK || echo NOPE

         # succeeds  


hotcopy, archive, transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. create mysqlhotcopy section in :file:`~/.my.cnf` ie `/root/.my.cnf` as this must be 
   run as root in order to have access to the mysql DB files

::

    [mysqlhotcopy]
    socket    = /var/lib/mysql/mysql.sock
    # if somehow the socket config is ignored by mysqlhotcopy then use option `--socket /tmp/mysql.sock`
    host      = localhost
    user      = root
    password  = ***
    database = information_schema
    # 
    # NB needs a database specified to allow DB connection to make the locks, 
    # but database to backup is provided as an argument to mysqlhotbackup 
    # mitigate the duplicity by using the system metadata databse `information_schema` 
    

The hotcopy is very fast compared to the tgz creation, these 
are done separated (not in a pipe for example) so the time the DB is locked is 
kept to a minimum::

    [root@belle7 blyth]# mysqlhotcopy.py tmp_offline_db hotcopy archive transfer
    2013-05-16 17:11:16,649 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py tmp_offline_db hotcopy archive transfer
    2013-05-16 17:11:16,653 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db 
    2013-05-16 17:11:16,673 env.mysqlhotcopy.mysqlhotcopy INFO     db size in MB 152.27 
    2013-05-16 17:11:16,673 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== hotcopy 
    2013-05-16 17:11:16,673 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 380.675 MB less than    free 497384.726562 MB 
    2013-05-16 17:11:16,673 env.mysqlhotcopy.mysqlhotcopy INFO     hotcopy of database tmp_offline_db into outd /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711 
    2013-05-16 17:11:16,685 env.mysqlhotcopy.mysqlhotcopy INFO     proceed with MySQLHotCopy /usr/bin/mysqlhotcopy tmp_offline_db /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711   
    2013-05-16 17:11:17,256 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_hotcopy': 0.58285903930664062} 
    2013-05-16 17:11:17,257 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== archive 
    2013-05-16 17:11:17,257 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 380.675 MB less than    free 497231.179688 MB 
    2013-05-16 17:11:17,257 env.mysqlhotcopy.mysqlhotcopy INFO     tagd /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711  into Tar /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz tmp_offline_db gz  
    2013-05-16 17:11:17,258 env.mysqlhotcopy.tar INFO     creating /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz from /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711/tmp_offline_db 
    2013-05-16 17:15:35,201 env.mysqlhotcopy.tar WARNING  deleting src /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711/tmp_offline_db directory following archive creation 
    2013-05-16 17:15:35,241 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_hotcopy': 0.58285903930664062, 'archive': 257.98302602767944, '_archive': 257.98317098617554} 
    2013-05-16 17:15:35,241 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== transfer 
    2013-05-16 17:15:35,241 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 380.675 MB less than    free 497335.757812 MB 
    2013-05-16 17:15:35,241 env.mysqlhotcopy.mysqlhotcopy INFO     transfer Tar /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz tmp_offline_db gz  to remotenode C   
    2013-05-16 17:15:35,242 env.mysqlhotcopy.common INFO     transfer /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz C  
    ssh C "mkdir -p /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db " 
    ssh: connect to host 140.112.101.190 port 22: Connection timed out
    time scp /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz C:/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz 
    ssh: connect to host 140.112.101.190 port 22: Connection timed out
    lost connection

    real    3m9.056s
    user    0m0.000s
    sys     0m0.007s
    2013-05-16 17:21:53,351 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'transfer': 378.10944199562073, '_hotcopy': 0.58285903930664062, '_transfer': 378.10959100723267, 'archive': 257.98302602767944, '_archive': 257.98317098617554} 
    [root@belle7 blyth]# 


When doing `archive`, `transfer` (or `extract`) separately from the `hotcopy` specifying the timestamp
is required as shown below.


extract dryrun
~~~~~~~~~~~~~~~~~

Due to the potential for damage from tampering with the mysql datadir, **extraction** requres a few options

Extraction of the `dybdb1.ihep.ac.cn` tarball onto belle7.  As no DB called `tmp_ligs_offline_db` 
exists on `belle7` it is necessary to provide the appropriate datadir for the node 
with `--containerdir /var/lib/mysql`

This is a dryrun due to option `-n` in order to check the paths are as expected:
::

    [root@belle7 ~]# mysqlhotcopy.py -t 20130522_1541 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 --containerdir /var/lib/mysql --ALLOWEXTRACT -n tmp_ligs_offline_db examine extract
    2013-05-23 12:01:37,782 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py -t 20130522_1541 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 --containerdir /var/lib/mysql --ALLOWEXTRACT -n tmp_ligs_offline_db examine extract
    2013-05-23 12:01:37,782 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db 
    2013-05-23 12:01:37,794 env.mysqlhotcopy.mysqlhotcopy INFO     failed to instanciate connection to database tmp_ligs_offline_db with exception Error 1049: Unknown database 'tmp_ligs_offline_db'  
    2013-05-23 12:01:37,794 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== examine 
    2013-05-23 12:01:37,794 env.mysqlhotcopy.tar INFO     examining /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz 
    2013-05-23 12:02:13,057 env.mysqlhotcopy.tar INFO     archive contains 7 items with commonprefix "" flattop True 
    2013-05-23 12:02:13,057 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 35.262603998184204, 'examine': 35.262594938278198} 
    2013-05-23 12:02:13,057 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== extract 
    2013-05-23 12:02:13,057 env.mysqlhotcopy.mysqlhotcopy WARNING  no valid db connection using static opts.mb_required 2000 
    2013-05-23 12:02:13,058 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 2000 MB less than    free 494499.9375 MB 
    2013-05-23 12:02:13,058 env.mysqlhotcopy.mysqlhotcopy INFO     proceeding
    2013-05-23 12:02:13,058 env.mysqlhotcopy.mysqlhotcopy INFO     extract Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz tmp_ligs_offline_db gz  into containerdir /var/lib/mysql   
    2013-05-23 12:02:13,058 env.mysqlhotcopy.tar INFO     _flat_extract opening tarfile /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz 
    2013-05-23 12:02:48,471 env.mysqlhotcopy.tar INFO     dryrun: _flat_extract into target /var/lib/mysql/tmp_ligs_offline_db_0 for 7 members with toplevelname tmp_ligs_offline_db_0 
    2013-05-23 12:02:48,472 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 35.262603998184204, 'examine': 35.262594938278198, 'extract': 35.413694858551025, '_extract': 35.414549112319946} 
    [root@belle7 ~]# 


actual extraction
~~~~~~~~~~~~~~~~~~

An interactive confirmation of **YES** is required before the extraction is done.

::

    [root@belle7 ~]# mysqlhotcopy.py -t 20130522_1541 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 --containerdir /var/lib/mysql --ALLOWEXTRACT  tmp_ligs_offline_db examine extract
    2013-05-23 12:06:33,546 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py -t 20130522_1541 --node dybdb1.ihep.ac.cn --rename tmp_ligs_offline_db_0 --containerdir /var/lib/mysql --ALLOWEXTRACT tmp_ligs_offline_db examine extract
    2013-05-23 12:06:33,546 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db 
    2013-05-23 12:06:33,561 env.mysqlhotcopy.mysqlhotcopy INFO     failed to instanciate connection to database tmp_ligs_offline_db with exception Error 1049: Unknown database 'tmp_ligs_offline_db'  
    2013-05-23 12:06:33,561 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== examine 
    2013-05-23 12:06:33,562 env.mysqlhotcopy.tar INFO     examining /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz 
    2013-05-23 12:07:08,913 env.mysqlhotcopy.tar INFO     archive contains 7 items with commonprefix "" flattop True 
    2013-05-23 12:07:08,913 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 35.351444005966187, 'examine': 35.35143518447876} 
    2013-05-23 12:07:08,913 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== extract 
    2013-05-23 12:07:08,914 env.mysqlhotcopy.mysqlhotcopy WARNING  no valid db connection using static opts.mb_required 2000 
    2013-05-23 12:07:08,914 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 2000 MB less than    free 494499.882812 MB 
    DO YOU REALLY WANT TO extract Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz tmp_ligs_offline_db gz  into containerdir /var/lib/mysql    ? ENTER "YES" TO PROCEED : YES
    2013-05-23 12:07:48,589 env.mysqlhotcopy.mysqlhotcopy INFO     proceeding
    2013-05-23 12:07:48,589 env.mysqlhotcopy.mysqlhotcopy INFO     extract Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz tmp_ligs_offline_db gz  into containerdir /var/lib/mysql   
    2013-05-23 12:07:48,589 env.mysqlhotcopy.tar INFO     _flat_extract opening tarfile /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130522_1541.tar.gz 
    2013-05-23 12:08:23,906 env.mysqlhotcopy.tar INFO     _flat_extract into target /var/lib/mysql/tmp_ligs_offline_db_0 for 7 members with toplevelname tmp_ligs_offline_db_0 
    2013-05-23 12:09:06,346 env.mysqlhotcopy.tar INFO     total 2429412
    -rw-rw---- 1 mysql mysql       8746 Feb  4 16:07 DqChannelStatus.frm
    -rw-rw---- 1 mysql mysql 1439608104 May 16 19:15 DqChannelStatus.MYD
    -rw-rw---- 1 mysql mysql 1024402432 May 16 19:42 DqChannelStatus.MYI
    -rw-rw---- 1 mysql mysql       8908 May 13 13:16 DqChannelStatusVld.frm
    -rw-rw---- 1 mysql mysql   17397375 May 20 06:26 DqChannelStatusVld.MYD
    -rw-rw---- 1 mysql mysql    3826688 May 20 06:26 DqChannelStatusVld.MYI

    2013-05-23 12:09:06,347 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_examine': 35.351444005966187, 'examine': 35.35143518447876, 'extract': 77.757769107818604, '_extract': 117.43390297889709} 
    [root@belle7 ~]# 



mysqlhotcopy options
----------------------

`--allowold`
   Move any existing version of the destination to a backup directory
   for the duration of the copy. If the copy successfully completes, the backup
   directory is deleted - unless the --keepold flag is set.  If the copy fails,
   the backup directory is restored.

   The backup directory name is the original name with "_old" appended.
   Any existing versions of the backup directory are deleted.


Size estimation 
-------------------

Size of hotcopy directory close to that estimated from DB, tgz is factor of 3 smaller::

    [blyth@belle7 DybPython]$ echo "select round(sum((data_length+index_length-data_free)/1024/1024),2) as TOT_MB from information_schema.tables where table_schema = 'tmp_offline_db' " | mysql -t 
    +--------+
    | TOT_MB |
    +--------+
    | 152.27 | 
    +--------+

    [blyth@belle7 mysqlhotcopy]$ sudo du -h 20130514_1832        
    154M    20130514_1832/tmp_offline_db
    154M    20130514_1832

    [blyth@belle7 mysqlhotcopy]$ sudo du -h 20130514_1832.tar.gz 
    49M     20130514_1832.tar.gz


Prepare directories on target for the transfer
-----------------------------------------------

::

    [blyth@cms01 dbbackup]$ sudo mkdir -p /data/var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/
    [blyth@cms01 dbbackup]$ sudo mkdir -p /data/var/dbbackup/mysqlhotcopy/dybdb2.ihep.ac.cn/tmp_ligs_offline_db/
    [blyth@cms01 dbbackup]$ sudo chown -R dayabayscp /data/var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/
    [blyth@cms01 dbbackup]$ sudo chown -R dayabayscp /data/var/dbbackup/mysqlhotcopy/dybdb2.ihep.ac.cn/tmp_ligs_offline_db/

And for testing N to H transfers::

    [blyth@hfag data]$ sudo mkdir -p /data/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db 


Table-by-table hotcopy, to minimise table lock time ?
-------------------------------------------------------

Whilst possible to handle DBI tables in separate hotcopies of payload+validity pairs 
using `--addtodest` option this might not be a consistent backup, and there is the LOCALSEQNO
too to cause problems.


`--addtodest`
   Don't rename target directory if it already exists, just add the copied files into it.

   This is most useful when backing up a database with many large tables and
   you don't want to have all the tables locked for the whole duration.

   In this situation, if you are happy for groups of tables to be backed up
   separately (and thus possibly not be logically consistant with one another)
   then you can run mysqlhotcopy several times on the same database each with
   different db_name./table_regex/.  All but the first should use the --addtodest
   option so the tables all end up in the same directory.



Behaviour with crashed tables
-------------------------------

* http://www.databasejournal.com/features/mysql/article.php/3300511/Repairing-Database-Corruption-in-MySQL.htm

::

	[root@dybdb1 mysqlhotcopy]#   ./mysqlhotcopy.py -l debug tmp_ligs_offline_db hotcopy archive  
	2013-05-20 11:15:01,291 __main__ INFO     ./mysqlhotcopy.py -l debug tmp_ligs_offline_db hotcopy archive
	2013-05-20 11:15:01,294 __main__ INFO     backupdir /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db 
	2013-05-20 11:15:01,311 __main__ INFO     db size in MB 3760.22 
	2013-05-20 11:15:01,312 __main__ INFO     ================================== hotcopy 
	2013-05-20 11:15:01,312 __main__ INFO     sufficient free space,      required 9400.55 MB less than    free 20069.015625 MB 
	2013-05-20 11:15:01,312 __main__ INFO     hotcopy of database tmp_ligs_offline_db into outd /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130520_1115 
	2013-05-20 11:15:01,333 __main__ INFO     proceed with MySQLHotCopy /usr/bin/mysqlhotcopy  tmp_ligs_offline_db /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130520_1115   
	DBD::mysql::db do failed: Table './tmp_ligs_offline_db/DqChannelStatus' is marked as crashed and should be repaired at /usr/bin/mysqlhotcopy line 467.
	2013-05-20 11:15:01,828 __main__ INFO     seconds {'_hotcopy': 0.51553797721862793} 
	2013-05-20 11:15:01,828 __main__ INFO     ================================== archive 
	2013-05-20 11:15:01,828 __main__ INFO     sufficient free space,      required 9400.55 MB less than    free 20069.0078125 MB 
	2013-05-20 11:15:01,828 __main__ INFO     tagd /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130520_1115  into Tar /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130520_1115.tar.gz tmp_ligs_offline_db gz  
	enter "YES" to confirm deletion of sourcedir /var/dbbackup/mysqlhotcopy/dybdb1.ihep.ac.cn/tmp_ligs_offline_db/20130520_1115 :YES
	2013-05-20 11:15:42,169 __main__ INFO     seconds {'_hotcopy': 0.51553797721862793, 'archive': 40.34028697013855, '_archive': 40.340435981750488}



TODO:
-------

#. tarball digest dna 
#. tarball purging 


"""

# keep this standalone, ie no DybPython.DB
import os, logging, sys, tarfile, time, shutil, platform, re
from datetime import datetime
import fsutils
from db import DB
from cmd import CommandLine
log = logging.getLogger(__name__)
from common import timing, seconds
from tar import Tar



class MySQLHotCopy(CommandLine):
    """
    """
    _exenames = ['mysqlhotcopy','mysqlhotcopy5']
    _cmd = "%(exepath)s %(socket)s %(database)s%(regex)s %(outd)s "


class HotBackup(object):
    datestamp_ptn = re.compile("^\d{8}_\d{4}$")

    def match_dateddir(self, dir):
        leaf = dir.split("/")[-1]
        return self.datestamp_ptn.match(leaf)

    def __init__(self, opts, db ):
        database = opts.database
        tagd = os.path.join(opts.backupdir, opts.tag ) 
        tgzp = os.path.join(opts.backupdir, "%s.tar.gz" % opts.tag )
        tar = Tar(tgzp, toplevelname=database, remoteprefix=opts.remoteprefix, remotenode=opts.remotenode, confirm=opts.confirm, moveaside=opts.moveaside, ALLOWCLOBBER=opts.ALLOWCLOBBER)
        pass
        self.database = database
        self.tagd = tagd                     # where hot copies are created

        if db:
            datadir = db("select @@datadir as datadir")[0]['datadir']
        else:
            datadir = None

        # where tarballs are extracted defaults to the mysql datadir
        if opts.containerdir is None:
            containerdir = datadir
        else:
            containerdir = opts.containerdir        
        pass
        self.containerdir = containerdir
        self.datadir = datadir           
        self.tar = tar
        self.opts = opts                     # getting peripheral things via opts is OK, but not good style for criticals
        self.db = db
        self.enough = None

    def _space(self):
        dir = self.opts.backupdir 
        if not os.path.exists(dir):
            log.info("creating backupdir %s " % dir )
            os.makedirs(dir)
        pass
        du = fsutils.disk_usage(dir)

        if not self.db:
            mb_required = self.opts.mb_required
            log.warn("no valid db connection using static opts.mb_required %s " % mb_required )
        else:
            mb_required = self.opts.sizefactor*self.db.size   

        mb_free = du['mb_free']
        enough = mb_free > mb_required 
        if enough:
            log.info("sufficient free space,      required %s MB less than    free %s MB " % (mb_required, mb_free))
        else:
            log.warn("insufficient free space,   required %s MB greater than free %s MB " % (mb_required, mb_free))
        self.enough = enough
        return enough 

    def __call__(self, verb):
        """
        :param verb: 
        """
        log.info("================================== %s " % verb )
        if verb == "hotcopy":
            self._hotcopy(self.database, self.tagd)
        elif verb == "coldcopy":
            self._coldcopy(self.database, self.tagd)
        elif verb == "space":
            self._space()
        elif verb == "archive":
            self._archive()
        elif verb == "examine":
            self._examine()
        elif verb == "extract":
            self._extract()
        elif verb == "transfer":
            self._transfer()
        elif verb == "ls":
            self._ls()
        elif verb == "purge":
            self._purge()
        else:
            log.warn("unhandled verb %s " % verb ) 
        log.info("seconds %s " % seconds )

    def _space_check(self):
        enough = self._space()
        if not enough:
            msg = "ABORT as not enough space" 
            log.fatal(msg)
            raise Exception(msg)
        else:
            pass

    def _purge(self):
        dir = self.opts.backupdir
        keep = self.opts.keep
        tgz_ptn = re.compile("^\d{8}_\d{4}\.tar\.gz$")
        tgzs = sorted(filter(lambda _:tgz_ptn.match(_), os.listdir(dir)), reverse=True)
        log.info("purging tarballs from %s keeping latest %s " % ( dir, keep )) 
        count = 0  
        for i, tgz in enumerate(tgzs):
            path = os.path.join(dir, tgz)
            if i + 1 > keep:
                act = "*purge*"
                os.remove(path) 
                count += 1
            else:
                act = ""
            log.info("%-4s %s %s " % ( i, path, act ))
            pass
        log.info("purged %s tarballs from %s " % ( count, dir )) 

    def _ls(self):
        print os.popen("ls -l %s" % self.opts.backupdir).read()

    def _transfer(self):
        """
        The path of the tar is assumed to be the same on the remote node
        """
        msg = "transfer %s to remotenode   " % (self.tar )
        if self.opts.dryrun:
            log.info("dryrun: " + msg )
            return 
        log.info(msg)
        self.tar.transfer() 
    _transfer = timing(_transfer)


    def _examine(self):
        """
        """
        self.tar.examine()
    _examine = timing(_examine)

    def _extract(self):
        """
        `self.containerdir` is normally the same as `self.datadir`
        """
        self._space_check()
        msg = "extract %s into containerdir %s   " % (self.tar, self.containerdir)
        if not self.opts.ALLOWEXTRACT:
            log.warn("extraction is not allowed without --ALLOWEXTRACT option, for protection ")
            return   
        pass

        if self.opts.confirm and not self.opts.dryrun:
            really = raw_input("DO YOU REALLY WANT TO %s ? ENTER \"YES\" TO PROCEED : "  % msg )
        else:
            really = "YES"
        pass
        if really == "YES":
            log.info("proceeding") 
            log.info(msg)
            self.tar.extract(self.containerdir, toplevelname=self.opts.rename, dryrun=self.opts.dryrun) 
        else:
            log.info("OK skipping [%s != YES] " % really )
        pass
    _extract = timing(_extract)

    def _archive(self):
        """
        `self.tagd` contains the database named directory  
        """
        self._space_check()
        msg = "tagd %s  into %s " % (self.tagd, self.tar) 
        if self.opts.dryrun:
            log.info("dryrun: " + msg )
            return 
        log.info(msg)
        self.tar.archive(self.tagd, self.opts.deleteafter, self.opts.flattop) 
    _archive = timing(_archive)

    def _hotcopy(self, database, outd ):
        """
        Makes sure the `outd` exists and is empty then invoke the hotcopy into it
        a sub-folder named after the database is created within the outd

        :param database:
        :param outd: the dated folder 
        """
        self._space_check()
        msg = "hotcopy of database %s into outd %s " % (database, outd) 
        if self.opts.dryrun:
             log.info("dryrun: " + msg )
             return 
        log.info(msg)

        if self.opts.regex:
            regex = self.opts.regex
        else:
            regex = ""

        if self.opts.socket:
             socket = "--socket=%s" % self.opts.socket
        else:
             socket = "" 

        cmd = MySQLHotCopy(database=database, outd=outd, regex=regex, socket=socket)
        self._ensure_empty_dateddir(outd)    
        log.info("proceed with %s " % cmd )
        cmd() 
    _hotcopy = timing(_hotcopy)

    def _ensure_empty_dateddir(self, outd):
        assert self.match_dateddir(outd), "outd %s leaf not a dated dir" % outd
        if os.path.exists(outd):
            shutil.rmtree(outd)
        os.makedirs(outd)

    def _coldcopy(self, database, outd):
        """
        :param database: name
        :param outd: dated output folder, which is emptied before doing the coldcopy 
        """
        self._space_check()
        datadir = self.datadir
        src = os.path.join(datadir, database)
        tgt = os.path.join(outd, database)
        msg = "coldcopy from src %s to tgt %s **without locking**   " % (src, tgt)
        if self.opts.dryrun:
            log.info("dryrun: " + msg )
            return 
        log.info(msg)
        self._ensure_empty_dateddir(outd) 

        if self.opts.regex == None:
            _ignore = None
        else:
            log.info("_coldcopy interpreting regex %s as python pattern " % self.opts.regex )
            ptn = re.compile(self.opts.regex)
            def _ignore(dir, names):
                log.info("_coldcopy copytree : from %s names : %s " % ( len(names), repr(names) ))
                ignored = filter(lambda name:not ptn.match(name), names)
                log.info("_coldcopy copytree :  %s ignored : %s " % ( len(ignored), repr(ignored) ))
                return ignored
            pass
        fsutils.copytree( src, tgt, ignore=_ignore ) 
        pass
    _coldcopy = timing(_coldcopy)



def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-l", "--loglevel", default="INFO" )
    op.add_option("-s", "--sect",  default = "mysqlhotcopy", help="name of config section in :file:`~/.my.cnf` " )
    op.add_option("-b", "--backupdir",   default = "/var/dbbackup/mysqlhotcopy/%(node)s/%(database)s", help="base directory under which hotcopy backup tarballs are arranged in dated folders. Default %default " )
    op.add_option(      "--node",  default = None, help="Node from which a tarball originates, when not specified is assumed to be invoking machine. Default %default " )
    op.add_option("-x", "--containerdir",  default = None, help="MySQL datadir under which folders for each database reside, when not specified is discerned from the DB. Default %default " )
    op.add_option("-z", "--sizefactor",  default = 2.5,  help="Scale factor between DB size estimate and free space demanded, 2.0 is agressive (3.0 should be safe) as remember need space for tarball as well as backupdir. Default %default " )
    op.add_option("-m", "--moveaside",  action="store_true",  help="When restoring and a preexisting database directory exists move it aside with a datestamp. If this is not selected the extract will abort. Default %default " )
    op.add_option("-D", "--nodeleteafter",  dest="deleteafter", default=True, action="store_false",  help="Normally directories are deleted after creation of archives, this option will inhibit the deletion. Default %default " )
    op.add_option(      "--regex",       default=None,  help="Regular expression string appended to dbname first argument of mysqlhotcopy used to include or exclude tables OR None to indicate all table. Default %default " )
    op.add_option("-r", "--remotenode",  default="C",  help="Remote node which the transfer command will scp the tarball to. Default %default " )
    op.add_option(      "--remoteprefix",  default="/data",  help="Prefix to tarball paths on remote node. Default %default " )
    op.add_option(      "--mb_required",  type=int, default=2000,  help="Free space requirement used when there is no valid DB connection, eg when extracting a DB onto a new node. Default %default " )
    op.add_option(      "--ALLOWEXTRACT",  action="store_true",  help="Avoid accidental extraction by requiring this option setting for this potentially destructive command. Default %default " )
    op.add_option(      "--ALLOWCLOBBER",  action="store_true",  help="Avoid accidental clobbering existing paths by requiring this option setting for extraction ontop of preexisting paths. Default %default " )
    op.add_option("-n", "--dryrun",  action="store_true",  help="Describe what will be done without doing it. Default %default " )
    op.add_option(      "--flattop",  action="store_true",  help="Use flat top structure for created archives, allowing toplevelname changes. Default %default " )
    op.add_option(      "--rename",  default=None,  help="Extract archive into `rename` directory withinn `containerdir`. Default %default " )
    op.add_option("-C", "--noconfirm", dest="confirm",  default=True, action="store_false",  help="Disable interactive confirmation of deletion of hotcopy folders. Default %default " )
    op.add_option("-t", "--tag", default=datetime.now().strftime("%Y%m%d_%H%M"), help="a string used to identify a backup directory and tarball. Defaults to current time string, %default " )
    op.add_option(      "--socket",  default = None,  help="Path to mysql socket used by mysqlhotcopy command, use this if fails to read from [mysqlhotcopy] section.  Default %default " )
    op.add_option("-k", "--keep",  default = 3,  help="Number of dated tarballs to retain within each backupdir when purging, ie for each node and database. Default %default " )
    opts, args = op.parse_args()
    assert len(args) > 1, "expect at least 2 arguments,  the first is database name followed by one or more command verbs"
    level=getattr(logging,opts.loglevel.upper()) 
   
    if opts.logpath:  # logs to file as well as console
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(opts.logformat)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)  # add the handler to the root logger
    else:
        try: 
            logging.basicConfig(format=opts.logformat,level=level)
        except TypeError:
            hdlr = logging.StreamHandler()
            formatter = logging.Formatter(opts.logformat)
            hdlr.setFormatter(formatter)
            log.addHandler(hdlr)
            log.setLevel(level)
        pass
    pass

    log.info(" ".join(sys.argv))

    database = args.pop(0)

    if opts.node is None:
        opts.node = platform.node()

    if '%' in opts.backupdir:
        opts.backupdir = opts.backupdir % dict(node=opts.node, database=database)
    log.info("backupdir %s " % opts.backupdir )


    try:
        db = DB(opts.sect, database="information_schema")   # use information_schema DB as that should always be present
    except Exception, e:
        log.info("failed to instanciate connection to database %s with exception %s " % (database, e))
        db = None

    opts.database = database

    return opts, args, db 


def main():    
    opts, args, db = parse_args_(__doc__)
    if db:
        log.info("db size in MB %s " % db.size )
    hb = HotBackup(opts, db)
    for verb in args: 
        hb(verb)


if __name__ == '__main__':
    main()



