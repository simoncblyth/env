#!/usr/bin/env python
"""
MySQL Hotcopy wrapper 
=======================

#. avoids filling disk by estimating space required for hotcopy, 
   from DB queries and file system free space checking 
#. creates tarballs in dated folders

TODO:

#. tarball digest dna 
#. tarball scp (experience suggests that is more reliable than scp on long term )
#. tarball purging 


Intended to be used in system python from sudo, operating from non-pristine 
env will cause errors related to setuptools.
Requires MySQLdb, check that and operating env with::

    sudo python -c "import MySQLdb"

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
    database  = tmp_offline_db
    host      = localhost
    user      = root
    password  = ***

The hotcopy is very fast compared to the tgz creation, these 
are done separated (not in a pipe for example) so the time the DB is locked is 
kept to a minimum::

    [blyth@belle7 scm]$ sudo python mysqlhotcopy.py 
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
import os, logging, sys, tarfile, time
from datetime import datetime
from fsutils import disk_usage
from db import DB
from cmd import CommandLine
log = logging.getLogger(__name__)

seconds = {}
def timing(func):
    def wrapper(*arg,**kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        global seconds
        seconds[func.func_name] = (t2-t1)
        return res 
    return wrapper

    
class MySQLHotCopy(CommandLine):
    """
    """
    _exenames = ['mysqlhotcopy','mysqlhotcopy5']
    _cmd = "%(exepath)s %(database)s %(outd)s "


class HotBackup(object):
    def __init__(self, opts ):
        self.opts = opts
        self.tag = datetime.now().strftime("%Y%m%d_%H%M")
        self.tagd = os.path.join(self.opts.base, self.tag ) 
        self.path = os.path.join(self.opts.base, "%s.tar.gz" % self.tag )

    def __call__(self):
        """
        """
        database = self.opts.database
        outd = self.tagd
        path = self.path

        self._hotcopy(database, outd)
        log.info("seconds %s " % seconds )
        self._make_tgz(outd, path )
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

    @timing
    def _make_tgz(self, outd , tgzpath):
        log.info("creating %s " % tgzpath )
        tgz = tarfile.open(tgzpath, "w:gz")
        tgz.add(outd, arcname="") 
        tgz.close() 
        os.rmdir(outd)


def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-l", "--loglevel", default="INFO" )
    op.add_option("-s", "--sect",  default = "mysqlhotcopy", help="name of config section in :file:`~/.my.cnf` " )
    op.add_option("-b", "--base",  default = "/var/scm/mysqlhotcopy", help="base directory under which hotcopy backup tarballs are arranged in dated folders" )
    opts, args = op.parse_args()

    level=getattr(logging,opts.loglevel.upper()) 
    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))

    db = DB(opts.sect)
    opts.database = db.dbc['database']
    return opts, args, db 


def main():    
    opts, args, db = parse_args_(__doc__)
    log.info("db size in MB %s " % db.size )
    mb_required = 2.0*db.size   
    du = disk_usage(opts.base)
    mb_free = du['mb_free']

    if mb_free < mb_required:
        log.warn("insufficient free space,   required %s MB greater than free %s MB " % (mb_required, mb_free))
    else:
        log.info("sufficient free space,      required %s MB less than    free %s MB " % (mb_required, mb_free))
        hb = HotBackup(opts)
        hb()


if __name__ == '__main__':
    main()



