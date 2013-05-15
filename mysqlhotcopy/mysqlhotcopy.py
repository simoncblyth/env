#!/usr/bin/env python
"""
MySQL Hotcopy wrapper 
=======================

#. avoids filling disk by estimating space required for hotcopy, 
   from DB queries and file system free space checking 
#. creates tarballs in dated folders

TODO:

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

  
Usage steps
-----------

hotcopy
~~~~~~~~~

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

    [root@belle7 mysqlhotcopy]# ./mysqlhotcopy.py tmp_offline_db hotcopy archive transfer

    2013-05-15 17:25:07,072 __main__ INFO     ./mysqlhotcopy.py tmp_offline_db hotcopy
    2013-05-15 17:25:07,095 __main__ INFO     db size in MB 152.27 
    2013-05-15 17:25:07,096 __main__ INFO     sufficient free space,      required 380.675 MB less than    free 497786.996094 MB 
    2013-05-15 17:25:07,096 __main__ INFO     hotcopy of database tmp_offline_db into outd /var/scm/mysqlhotcopy/20130515_1725 
    2013-05-15 17:25:07,100 __main__ INFO     proceed with MySQLHotCopy /usr/bin/mysqlhotcopy tmp_offline_db /var/scm/mysqlhotcopy/20130515_1725   
    2013-05-15 17:25:07,637 __main__ INFO     tagd /var/scm/mysqlhotcopy/20130515_1725  into Tar /var/scm/mysqlhotcopy/20130515_1725.tar.gz tmp_offline_db gz  
    2013-05-15 17:25:07,637 tar INFO     creating /var/scm/mysqlhotcopy/20130515_1725.tar.gz from /var/scm/mysqlhotcopy/20130515_1725/tmp_offline_db 
    2013-05-15 17:29:25,943 __main__ INFO     seconds {'_hotcopy': 0.54047489166259766, 'archive': 258.30557799339294, '_archive': 258.30591607093811} 
    [root@belle7 mysqlhotcopy]# 


When doing `archive`, `transfer` or `extract` separately from the `hotcopy` specifying the timestamp
is required as shown below.


extract
~~~~~~~~


::

    [root@belle7 mysqlhotcopy]# ./mysqlhotcopy.py -m -t 20130515_1725 tmp_offline_db extract
    2013-05-15 17:46:26,870 __main__ INFO     ./mysqlhotcopy.py -m -t 20130515_1725 tmp_offline_db extract
    2013-05-15 17:46:26,889 __main__ INFO     db size in MB 152.27 
    2013-05-15 17:46:26,889 __main__ INFO     sufficient free space,      required 380.675 MB less than    free 497584.421875 MB 
    2013-05-15 17:46:26,889 __main__ INFO     extract Tar /var/scm/mysqlhotcopy/20130515_1725.tar.gz tmp_offline_db gz  into extractdir /var/lib/mysql   
    2013-05-15 17:46:26,890 tar WARNING  moving aside pre-existing tgt dir /var/lib/mysql/tmp_offline_db to /var/lib/mysql/tmp_offline_db_20130515_174626 
    2013-05-15 17:46:26,890 tar INFO     extracting /var/scm/mysqlhotcopy/20130515_1725.tar.gz with toplevelname tmp_offline_db into extractdir /var/lib/mysql 
    2013-05-15 17:46:32,249 __main__ INFO     seconds {'_extract': 5.3598589897155762, 'extract': 5.3596851825714111} 
    [root@belle7 mysqlhotcopy]# 


Any preexisting DB is moved aside::

    mysql> show tables ;
    +------------------------------------------+
    | Tables_in_tmp_offline_db_20130515_174626 |
    +------------------------------------------+
    | CableMap                                 | 
    | CableMapVld                              | 
    | CalibPmtFineGain                         | 


Issues
-------

mysqlhotcopy does low level file copying, making version closeness important  

::

   dybdb1.ihep.ac.cn        5.0.45-community-log MySQL Community Edition (GPL)
   belle7.nuu.edu.tw        5.0.77-log Source distribution
   cms01.phys.ntu.edu.tw    4.1.22-log


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


"""

# keep this standalone, ie no DybPython.DB
import os, logging, sys, tarfile, time, shutil, platform
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
    verbs = "hotcopy archive extract transfer".split()
    def __init__(self, opts, db ):
        database = opts.database
        tagd = os.path.join(opts.backupdir, opts.tag ) 
        tgzp = os.path.join(opts.backupdir, "%s.tar.gz" % opts.tag )
        tar = Tar(tgzp, toplevelname=database)
        pass
        self.database = database
        self.tagd = tagd                     # where hot copies are created
        self.extractdir = opts.extractdir    # where tarballs are extracted
        self.tar = tar
        self.opts = opts                     # getting peripheral things via opts is OK, but not good style for criticals
        self.db = db

    def enoughspace(self):
        dir = self.opts.backupdir 
        if not os.path.exists(dir):
            log.info("creating backupdir %s " % dir )
            os.makedirs(dir)
        pass
        du = disk_usage(dir)
        mb_required = self.opts.sizefactor*self.db.size   
        mb_free = du['mb_free']
        enough = mb_free > mb_required 
        if enough:
            log.info("sufficient free space,      required %s MB less than    free %s MB " % (mb_required, mb_free))
        else:
            log.warn("insufficient free space,   required %s MB greater than free %s MB " % (mb_required, mb_free))
        return enough 

    def __call__(self, verb):
        """
        :param verb: 
        """
        log.info("================================== %s " % verb )

        enough = self.enoughspace()
        if not enough:
            log.warn("ABORT %s as not enough space" % verb )
            return 
        if verb == "hotcopy":
            self._hotcopy(self.database, self.tagd)
        elif verb == "archive":
            self._archive()
        elif verb == "extract":
            self._extract()
        elif verb == "transfer":
            self._transfer()
        else:
            log.warn("unhandled verb %s " % verb ) 
        log.info("seconds %s " % seconds )


    @timing
    def _transfer(self):
        """
        The path of the tar is assumed to be the same on the remote node
        """
        msg = "transfer %s to remotenode %s   " % (self.tar, self.opts.remotenode )
        if self.opts.dryrun:
            log.info("dryrun: " + msg )
            return 
        log.info(msg)
        self.tar.transfer(self.opts.remotenode) 


    @timing
    def _extract(self):
        """
        `self.extractdir` contains the database named directory  
        """
        msg = "extract %s into extractdir %s   " % (self.tar, self.extractdir)
        if self.opts.dryrun:
            log.info("dryrun: " + msg )
            return 
        log.info(msg)
        self.tar.extract(self.extractdir, self.opts.moveaside) 

    @timing
    def _archive(self):
        """
        `self.tagd` contains the database named directory  
        """
        msg = "tagd %s  into %s " % (self.tagd, self.tar) 
        if self.opts.dryrun:
            log.info("dryrun: " + msg )
            return 
        log.info(msg)
        self.tar.archive(self.tagd, self.opts.deleteafter) 

    @timing
    def _hotcopy(self, database, outd ):
        """
        Makes sure the `outd` exists and is empty then invoke the hotcopy into it
        a sub-folder named after the database is created within the outd

        :param database:
        :param outd:
        """
        msg = "hotcopy of database %s into outd %s " % (database, outd) 
        if self.opts.dryrun:
             log.info("dryrun: " + msg )
             return 
        log.info(msg)
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
    op.add_option("-b", "--backupdir",   default = "/var/dbbackup/mysqlhotcopy/%(node)s/%(database)s", help="base directory under which hotcopy backup tarballs are arranged in dated folders. Default %default " )
    op.add_option("-x", "--extractdir",  default = "/var/lib/mysql", help="MySQL data dir under which folders for each database reside, Default %default " )
    op.add_option("-z", "--sizefactor",  default = 2.5,  help="Scale factor between DB size estimate and free space demanded, 2.0 is agressive (3.0 should be safe) as remember need space for tarball as well as backupdir. Default %default " )
    op.add_option("-m", "--moveaside",  action="store_true",  help="When restoring and a preexisting database directory exists move it aside with a datestamp. If this is not selected the extract will abort. Default %default " )
    op.add_option("-D", "--nodeleteafter",  dest="deleteafter", default=True, action="store_false",  help="Normally directories are deleted after creation of archives, this option will inhibit the deletion. Default %default " )
    op.add_option("-r", "--remotenode",  default="C",  help="Remote node which the transfer command will scp the tarball to. Default %default " )
    op.add_option("-n", "--dryrun",  action="store_true",  help="Describe what will be done without doing it. Default %default " )
    op.add_option("-t", "--tag", default=datetime.now().strftime("%Y%m%d_%H%M"), help="a string used to identify a backup directory and tarball. Defaults to current time string, %default " )
    opts, args = op.parse_args()
    assert len(args) > 1, "expect at least 2 arguments,  the first is database name followed by one or more command verbs"
    level=getattr(logging,opts.loglevel.upper()) 
    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))

    database = args.pop(0)
    if '%' in opts.backupdir:
        opts.backupdir = opts.backupdir % dict(node=platform.node(), database=database)
    log.info("backupdir %s " % opts.backupdir )

    db = DB(opts.sect, database=database)
    opts.database = database

    return opts, args, db 


def main():    
    opts, args, db = parse_args_(__doc__)
    log.info("db size in MB %s " % db.size )
    hb = HotBackup(opts, db)
    for verb in args: 
        hb(verb)


if __name__ == '__main__':
    main()



