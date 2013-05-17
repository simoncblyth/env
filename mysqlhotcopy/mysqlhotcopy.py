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


Commands
---------

The first argument of the `mysqlhotcopy.py` script specifies the name of a mysql database
to operate upon.  Subsequent arguments specify actions to take. Order is important. 


`hotcopy`
      use *mysqlhotcopy* to copy the mysql datadir for a single database into a dated folder under `backupdir`

`archive`
      archives the dated folder into a tarball and deletes the dated folder

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

          # using regex to only include 2 tables  

    ./mysqlhotcopy.py --regex './^\(DqChannelPacked\|DqChannelPackedVld\)/'  tmp_offline_db hotcopy       

          # have to escape the brackets and pipe symbol to protect from shell interpretation

    ./mysqlhotcopy.py -C --regex './^LOCALSEQNO/' tmp_offline_db hotcopy archive 

          # for quick machinery testing, restrict to just handling a small table and disable interactive confirmations
          # note that hotcopy will delete a pre-existing same minute folder however


hotcopy, archive, transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


extract
~~~~~~~~

Due to the potential for damage from tampering with the mysql datadir, **extraction** requres a few options
and interactive confirmation.

::

    [root@belle7 blyth]# mysqlhotcopy.py --moveaside --ALLOWEXTRACT -t 20130516_1711 tmp_offline_db extract
    2013-05-16 17:50:56,837 env.mysqlhotcopy.mysqlhotcopy INFO     /home/blyth/env/bin/mysqlhotcopy.py --moveaside --ALLOWEXTRACT -t 20130516_1711 tmp_offline_db extract
    2013-05-16 17:50:56,841 env.mysqlhotcopy.mysqlhotcopy INFO     backupdir /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db 
    2013-05-16 17:50:56,865 env.mysqlhotcopy.mysqlhotcopy INFO     db size in MB 152.27 
    2013-05-16 17:50:56,865 env.mysqlhotcopy.mysqlhotcopy INFO     ================================== extract 
    2013-05-16 17:50:56,866 env.mysqlhotcopy.mysqlhotcopy INFO     sufficient free space,      required 380.675 MB less than    free 496874.898438 MB 
    DO YOU REALLY WANT TO extract Tar /var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130516_1711.tar.gz tmp_offline_db gz  into extractdir /var/lib/mysql    ? ENTER "YES" TO PROCEED : 
    2013-05-16 17:50:59,271 env.mysqlhotcopy.mysqlhotcopy INFO     OK skipping [ != YES] 
    2013-05-16 17:50:59,271 env.mysqlhotcopy.mysqlhotcopy INFO     seconds {'_extract': 2.404871940612793} 
    [root@belle7 blyth]# 

Any preexisting DB is moved aside::

    mysql> show tables ;
    +------------------------------------------+
    | Tables_in_tmp_offline_db_20130515_174626 |
    +------------------------------------------+
    | CableMap                                 | 
    | CableMapVld                              | 
    | CalibPmtFineGain                         | 




Possibilities
--------------

MySQL `datadir` introspection::

    mysql> select @@datadir as datadir ;
    +-----------------+
    | datadir         |
    +-----------------+
    | /var/lib/mysql/ | 
    +-----------------+
    1 row in set (0.00 sec)


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


TODO:
-------

#. tarball digest dna 
#. tarball purging 


"""

# keep this standalone, ie no DybPython.DB
import os, logging, sys, tarfile, time, shutil, platform, re
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
    _cmd = "%(exepath)s %(database)s%(regex)s %(outd)s "


class HotBackup(object):
    verbs = "hotcopy archive extract transfer ls purge".split()
    def __init__(self, opts, db ):
        database = opts.database
        tagd = os.path.join(opts.backupdir, opts.tag ) 
        tgzp = os.path.join(opts.backupdir, "%s.tar.gz" % opts.tag )
        tar = Tar(tgzp, toplevelname=database, remoteprefix=opts.remoteprefix, remotenode=opts.remotenode, confirm=opts.confirm)
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
        elif verb == "ls":
            self._ls()
        elif verb == "purge":
            self._purge()
        else:
            log.warn("unhandled verb %s " % verb ) 
        log.info("seconds %s " % seconds )

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

    def _extract(self):
        """
        `self.extractdir` contains the database named directory  
        """
        msg = "extract %s into extractdir %s   " % (self.tar, self.extractdir)
        if not self.opts.ALLOWEXTRACT:
            log.warn("extraction is not allowed without --ALLOWEXTRACT option, for protection ")
            return   
        if self.opts.dryrun:
            log.info("dryrun: " + msg )
            return 
        pass
        really = raw_input("DO YOU REALLY WANT TO %s ? ENTER \"YES\" TO PROCEED : "  % msg )
        if really == "YES":
            log.info("proceeding") 
            log.info(msg)
            self.tar.extract(self.extractdir, self.opts.moveaside) 
        else:
            log.info("OK skipping [%s != YES] " % really )
        pass
    _extract = timing(_extract)

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
    _archive = timing(_archive)

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

        if self.opts.regex:
            regex = self.opts.regex
        else:
            regex = ""

        cmd = MySQLHotCopy(database=database, outd=outd, regex=regex)
        if os.path.exists(outd):
            os.rmdir(outd)
        os.makedirs(outd)
        log.info("proceed with %s " % cmd )
        cmd() 
    _hotcopy = timing(_hotcopy)









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
    op.add_option(      "--regex",       default=None,  help="Regular expression string appended to dbname first argument of mysqlhotcopy used to include or exclude tables OR None to indicate all table. Default %default " )
    op.add_option("-r", "--remotenode",  default="C",  help="Remote node which the transfer command will scp the tarball to. Default %default " )
    op.add_option(      "--remoteprefix",  default="/data",  help="Prefix to tarball paths on remote node. Default %default " )
    op.add_option(      "--ALLOWEXTRACT",  action="store_true",  help="Avoid accidental extraction by requiring this option setting for this potentially destructive command. Default %default " )
    op.add_option("-n", "--dryrun",  action="store_true",  help="Describe what will be done without doing it. Default %default " )
    op.add_option("-C", "--noconfirm", dest="confirm",  default=True, action="store_false",  help="Disable interactive confirmation of deletion of hotcopy folders. Default %default " )
    op.add_option("-t", "--tag", default=datetime.now().strftime("%Y%m%d_%H%M"), help="a string used to identify a backup directory and tarball. Defaults to current time string, %default " )
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



