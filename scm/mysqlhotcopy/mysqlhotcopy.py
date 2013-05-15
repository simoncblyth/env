#!/usr/bin/env python
"""
MySQL Hotcopy wrapper 
=======================

#. avoids filling disk by estimating space required for hotcopy, 
   from DB queries and file system free space checking 
#. creates tarballs in dated folders

TODO:

#. recovery feature

#. offboxing 

    #. tarball digest dna 
    #. tarball scp (experience suggests that is more reliable than rsync for long term usage )
    #. tarball purging 

Intended to be used in system python from sudo, operating from non-pristine 
env will cause errors related to setuptools.
Requires MySQLdb, check that and operating env with::

    sudo python -c "import MySQLdb"


mysqlhotcopy options
----------------------

`--allowold`
           Move any existing version of the destination to a backup directory
           for the duration of the copy. If the copy successfully completes, the backup
           directory is deleted - unless the --keepold flag is set.  If the copy fails,
           the backup directory is restored.

           The backup directory name is the original name with "_old" appended.
           Any existing versions of the backup directory are deleted.

Issues
-------

mysqlhotcopy does low level file copying, making version closeness important  

::

   dybdb1.ihep.ac.cn        5.0.45-community-log MySQL Community Edition (GPL)
   belle7.nuu.edu.tw        5.0.77-log Source distribution
   cms01.phys.ntu.edu.tw    4.1.22-log
   
Usage steps
-----------

#. create mysqlhotcopy section in :file:`~/.my.cnf` ie `/root/.my.cnf` as this must be 
   run as root in order to have access to the mysql DB files

::

    [mysqlhotcopy]
    socket    = /var/lib/mysql/mysql.sock
    host      = localhost
    user      = root
    password  = ***
    database = information_schema
    # 
    # NB needs a database specified to allow DB connection to make the locks, 
    # but database to backup is provided as an argument to mysqlhotbackup 
    # mitigate the duplicity by using the system metadata `information_schema` 
    

The hotcopy is very fast compared to the tgz creation, these 
are done separated (not in a pipe for example) so the time the DB is locked is 
kept to a minimum::

    [blyth@belle7 scm]$ sudo python mysqlhotcopy.py tmp_offline_db hotcopy
    2013-05-14 18:32:12,881 __main__ INFO     mysqlhotcopy.py tmp_offline_db
    2013-05-14 18:32:12,884 __main__ INFO     proceed with MySQLHotCopy /usr/bin/mysqlhotcopy tmp_offline_db /var/scm/mysqlhotcopy/20130514_1832   
    2013-05-14 18:32:13,442 __main__ INFO     seconds {'_hotcopy': 0.560593843460083} 
    2013-05-14 18:32:13,443 __main__ INFO     creating /var/scm/mysqlhotcopy/20130514_1832.tar.gz 
    2013-05-14 18:36:31,429 __main__ INFO     seconds {'_make_tgz': 257.9861190319061, '_hotcopy': 0.560593843460083} 
    [blyth@belle7 scm]$ 

    [blyth@belle7 mysqlhotcopy]$ sudo tar ztvf 20130514_1832.tar.gz
    drwxr-xr-x root/root         0 2013-05-14 18:32:12 /
    drwxr-x--- mysql/mysql       0 2013-05-14 18:32:13 tmp_offline_db/
    -rw-rw---- mysql/mysql       0 2013-04-30 18:28:16 tmp_offline_db/SupernovaTrigger.MYD
    -rw-rw---- mysql/mysql    8908 2012-08-17 20:06:30 tmp_offline_db/CalibPmtFineGainVld.frm
    -rw-rw---- mysql/mysql 3119296 2012-08-17 20:06:34 tmp_offline_db/HardwareID.MYD
    -rw-rw---- mysql/mysql    1024 2013-04-30 18:28:16 tmp_offline_db/SupernovaTrigger.MYI
    -rw-rw---- mysql/mysql 14858000 2013-05-11 20:18:46 tmp_offline_db/DqChannelPacked.MYD
    -rw-rw---- mysql/mysql      561 2012-11-20 14:26:31 tmp_offline_db/DemoVld.MYD
    ...


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


"""

# keep this standalone, ie no DybPython.DB
import os, logging, sys, tarfile, time, shutil
from datetime import datetime
from fsutils import disk_usage
from db import DB
from cmd import CommandLine
log = logging.getLogger(__name__)
from common import timing, seconds
from tar import Tar

class MySQLHotCopy(CommandLine):
    """
    """
    _exenames = ['mysqlhotcopy','mysqlhotcopy5']
    _cmd = "%(exepath)s %(database)s %(outd)s "


class HotBackup(object):
    verbs = "hotcopy restore".split()
    def __init__(self, opts ):
        self.database = opts.database
        self.tagd = os.path.join(opts.backupdir, opts.tag ) 
        self.path = os.path.join(opts.backupdir, "%s.tar.gz" % opts.tag )
        self.restoredir = opts.restoredir

    def __call__(self, verb):
        """
        :param verb: 
        """
        if verb == "hotcopy":
            self.hotcopy()
        elif verb == "restore":
            self.restore()
        else:
            log.warn("unhandled verb %s " % verb ) 

    def restore(self):
        """
        """
        tf = Tar(self.path)
        tf.extract(self.tagd, topleveldir=self.database) 
        log.info("seconds %s " % seconds )

    def hotcopy(self):
        """
        """ 
        self._hotcopy(self.database, self.tagd)
        log.info("seconds %s " % seconds )
        tf = Tar(self.path)
        tf.create(self.tagd)  # self.tagd contains the database named directory  
        log.info("seconds %s " % seconds )

    @timing
    def _hotcopy(self, database, outd ):
        """
        Make sure the `outd` exists and is empty then invoke the hotcopy into it
        a sub-folder named after the database is created within the outd

        :param database:
        :param outd:
        """
        cmd = MySQLHotCopy(database=database, outd=outd)
        if os.path.exists(outd):
            os.rmdir(outd)
        os.makedirs(outd)
        log.info("proceed with %s " % cmd )
        cmd() 

def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-l", "--loglevel", default="INFO" )
    op.add_option("-s", "--sect",  default = "mysqlhotcopy", help="name of config section in :file:`~/.my.cnf` " )
    op.add_option("-b", "--backupdir",   default = "/var/scm/mysqlhotcopy", help="base directory under which hotcopy backup tarballs are arranged in dated folders. Default %default " )
    op.add_option("-r", "--restoredir",  default = "/var/mysql/lib", help="MySQL data dir under which folders for each database reside, Default %default " )
    op.add_option("-z", "--sizefactor",  default = 2.5,  help="Scale factor between DB size estimate and free space demanded, 2.0 is agressive (3.0 should be safe) as remember need space for tarball as well as backupdir. Default %default " )
    op.add_option("-t", "--tag", default=datetime.now().strftime("%Y%m%d_%H%M"), help="a string used to identify a backup directory and tarball. Defaults to current time string, %default " )
    opts, args = op.parse_args()

    level=getattr(logging,opts.loglevel.upper()) 
    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))

    assert len(args) == 2, "expect 2 arguments with database name and command verb "
    database, verb = args
    allowed = HotBackup.verbs
    assert verb in allowed, "verb %s is not one if the allowed %s " % ( verb, repr(allowed))
    db = DB(opts.sect, database=database)
    opts.database = database
    return opts, verb, db 


def main():    
    opts, verb, db = parse_args_(__doc__)
    log.info("db size in MB %s " % db.size )
    mb_required = opts.sizefactor*db.size   
    du = disk_usage(opts.backupdir)
    mb_free = du['mb_free']

    if mb_free < mb_required:
        log.warn("insufficient free space,   required %s MB greater than free %s MB " % (mb_required, mb_free))
    else:
        log.info("sufficient free space,      required %s MB less than    free %s MB " % (mb_required, mb_free))
        hb = HotBackup(opts)
        hb(verb)


if __name__ == '__main__':
    main()



