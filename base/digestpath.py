#!/usr/bin/env python
"""
Emits to stdout the hexdigest for file path provided, using chunked reading to avoid memory 
issues with large files. Any logging goes to stderr to avoid messing stdout.

Usage::

    ~/e/base/digestpath.py /path/to/file/to/digest 
    ~/e/base/digestpath.py /data/var/scm/backup '*.tar.gz' './dayabay' 
    ~/e/base/digestpath.py /data/var/scm/backup/dayabay '*.tar.gz' './repos/'
    ~/e/base/digestpath.py /data/var/scm/backup/dayabay '*.tar.gz' './tracs/'
    ~/e/base/digestpath.py /data/var/scm/backup/dayabay '*.tar.gz' './svn/'

Allows subsequnt access with::

    ~/e/base/digestpath.py /data/var/scm/backup/dayabay '*.tar.gz' './svn/' >  out.p
    d = eval(file("out.p").read())

http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

Typical usage, need to find and digest a load of tarballs and record results into a dict that 
can be transferred and compared at other end of transfer/or at a later time 

There is a python version change problem in comparisons, with some pythons emitting the size with an L
and some not::

	< {'dig': 'f2191c11e0304b1cd52c19f970bf8a83', 'size': 8947425}
	---
	> {'dig': 'f2191c11e0304b1cd52c19f970bf8a83', 'size': 8947425L}
	=== scm-backup-dnachecktgzs : FAIL /data/var/scm/backup/cms02/tracs/heprez/2012/09/17/123022/heprez.tar.gz


History
---------

CQ Backup failure incident discovered 2014-10-13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In response to Gaosongs request to handover  
maintenance of partitioned CQDB backup to him, I checked 
status.
As reminder of where backups are and what is checked look into the monitoring 
config section on cms01 ~/.env.cnf (it is necessary to remember the node)::

    [dbsrvmon]

    note = currently set to fail via age of 1 days,  60*60*24 = 86400 
    dbpath = ~/.env/dbsrvmon.sqlite
    chdir = /data/var/dbbackup/dbsrv/dybdb2.ihep.ac.cn/channelquality_db_dybdb2/archive/10000
    tn = channelquality_db_10000
    cmd = digestpath.py 
    valmon_version = 0.2 
    return = dict
    constraints = ( tarball_count >= 31, dna_mismatch == 0, age < 86400, )


Checking the chdir directory shows the last partitioned backup tarball 
is 3 months old::

    /data/var/dbbackup/dbsrv/dybdb2.ihep.ac.cn/channelquality_db_dybdb2/archive/10000/10000_last.tar.gz
    Jul 25 16:03 

Although the partitions are sized for full ones to be transferred only every 
10 days or so, the remainder tarball is transferred every day : supposedly to 
provide warnings of machinery issues.

Checking why the monitoring did not send email to me regarding this, do 
a manual monitor run on C. Find the digestpath.py to be failing with 
a find::

        [blyth@cms01 10000]$ find -L . -name '*.tar.gz'
        find: invalid predicate `-L'

* https://bitbucket.org/simoncblyth/env/commits/81264c3ac057

Jimmy added the "-L" predicate that fails on the version of 
find on cms01, at some later date env was updated on cms01
and backup monitoring started to be ineffective from 
soon after 2014-02-24.



"""
# dont use logging/argparse/optparse as want to stay ancient python compatible 
import os, sys, time, stat
try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5


class Logger(object):
   """Dummy logger""" 
   def emit(self, *args):sys.stderr.write(args[0] % tuple(args[1:]) + '\n') 
   info = emit
   warn = emit
   debug = emit 
   fatal = emit 

log = Logger()


def dnapath( path , times=False ):
    """
    :param path: 

    called by ``scm-backup-dna`` to drop the tarball `.dna` sidecars
    """
    t0 = time.time()
    
    hash = md5()
    size = 64*128   # 8192
    st = os.stat(path)
    sz = st[stat.ST_SIZE]
    f = open(path,'rb') 
    for chunk in iter(lambda: f.read(size), ''): 
        hash.update(chunk)
    f.close()
    dig = hash.hexdigest()
    dna = dict(dig=dig,size=int(sz))

    t1 = time.time()
    if times:dna['t'] = t1 - t0   ## not standardly part of dna, as will change 
    return dna

def dnatree( top, ptn , start, findoption="-L"):
    """
    :param top: directory to perform find from 
    :param ptn: find pattern eg '*.tar.gz'
    :param start: string start for relative paths, use "." for all or "./dayabay" to restrict 

    """ 
    d = {}
    cmd = "cd %(top)s ; find %(findoption)s -name '%(ptn)s' " % locals() 
    pipe = os.popen(cmd)
    for line in pipe.readlines():
        if line.startswith(start):
            name = line.strip()
            path = os.path.abspath(os.path.join( top, name ))
            d[name] = dnapath( path )
            #sys.stderr.write("%s:%s" % ( name, d[name] ) )

    rc = pipe.close()
    if not rc is None:
        rc = os.WEXITSTATUS(rc)
        assert rc, "error %s doing cmd: %s " % (rc, cmd)
 
    return d 


def read_dna(path, sidecar_ext=".dna"):
    dnap = path + sidecar_ext
    if os.path.exists(dnap):
        sdna = open(dnap,"r").read().strip()
        assert sdna[0] == '{' and sdna[-1] == '}', sdna
        rdna = eval(sdna)
    else:
        rdna = None
    return rdna

def check_tarball_dna(top, ptn, start, verbose=False):
    """
    Lack of a DNA sidecar is indicative of the transfer 
    still be in progress or failed.

    :return: summary dict to stdout


    `tarball_count`
                   Number of tarballs 
    `dna_missing`
                   Number of tarballs without DNA sidecar 
    `dna_match`
                   Number of tarballs which recalculated digest matching the sidecar 
    `dna_mismatch`
                   Number of tarballs which recalculated digest not matching the sidecar 

    `lookstamp`
                   Epoch timestamp when the check was made
    `lastchange`
                   Timestamp on the last changed tarball
    `age`
                   Age in seconds of the last changed tarball

    """
    smry = dict(dna_match=0,tarball_count=0,dna_mismatch=0,dna_missing=0)
    look = time.time() 
    ctime = {}
    dt = dnatree(top, ptn, start, findoption="")
    #log.info("dnatree top %s ptn %s start %s #dt %s " % (top,ptn,start,len(dt))) 

    for path, dna in dt.items():
        smry['tarball_count'] += 1
        ctime[path] = os.path.getctime(path)
        rdna = read_dna(path)
        if verbose:
            sys.stderr.write("%-30s %s %s " % ( path, dna, rdna ))
        if rdna is None:
            smry['dna_missing'] += 1    
        else:
            if dna == rdna:
                smry['dna_match'] +=1  
            else:
                smry['dna_mismatch'] +=1  
            pass
        pass
    pass
    smry['lookstamp']  = look
    smry['lastchange'] = max(ctime.values())
    smry['age'] = smry['lookstamp'] - smry['lastchange']
    return smry  

def main():
    narg = len(sys.argv)
    if narg == 1:
        print check_tarball_dna(os.getcwd(), '*.tar.gz','.')  
    elif narg == 2:
        print dnapath(sys.argv[1])
    elif narg == 4: 
        print dnatree(*sys.argv[1:])

if __name__ == '__main__':
    main()

