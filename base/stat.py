#!/usr/bin/env python
"""

From ``man -s 2 stat``

   The field st_atime is changed by file accesses, 
   e.g. by execve(2), mknod(2), pipe(2), utime(2) and read(2) (of more than zero bytes). 
   Other routines, like mmap(2), may or may not update st_atime.

   The field st_mtime is changed by file modifications, 
   e.g. by mknod(2), truncate(2), utime(2) and write(2) (of more than zero bytes).  
   Moreover, st_mtime of a directory is changed by the creation or deletion of files in that directory.  
   The st_mtime field is not changed for changes in owner, group, hard link count, or mode.

   The field st_ctime is changed by writing or by setting inode information (i.e., owner, group, link count, mode, etc.).


Translation:

atime 
    accessed, read or written to
mtime 
    actual contents modified
ctime 
    inode information (permissions, name, etc., the metadata, as it were) modified



Monitor file growth using stat, eg for checking rsync transfer of tarballs, usage::

	[blyth@cms01 e]$ python ~/e/base/stat.py  '/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/.dybaux-5088.tar.gz.B3FMx6'
	seconds 13026.0 delta 3:37:06       1435762688 bytes    1435.763 MB        0.110 MBps (during transfer) 
	[blyth@cms01 e]$ 

Commandline stat shows::

	[blyth@cms01 e]$ stat  '/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/.dybaux-5088.tar.gz.B3FMx6';date
	  File: `/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/.dybaux-5088.tar.gz.B3FMx6'
	  Size: 1452015616      Blocks: 2838760    IO Block: 4096   regular file
	Device: 306h/774d       Inode: 509413      Links: 1
	Access: (0600/-rw-------)  Uid: (  506/dayabayscp)   Gid: (  506/dayabayscp)
	Access: 2011-10-21 15:32:42.000000000 +0800
	Modify: 2011-10-21 19:11:14.000000000 +0800
	Change: 2011-10-21 19:11:14.000000000 +0800
	Fri Oct 21 19:11:16 CST 2011

During a transfer observe that **for the arriving tarball**: 

#. "Access" corresponds to creation (when the rsync transfer started)
#. "Modify" = "Change"  corresponds to last update, usually slightly earlier than current time 

Shortly after completion, a rename of rsync tmporary name has happened::

	[blyth@cms01 home]$ python ~/e/base/stat.py  '/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/dybaux-5088.tar.gz'
	seconds 17252.0 delta 4:47:32       1915045602 bytes    1915.046 MB        0.111 MBps (during transfer) 

	[blyth@cms01 home]$ stat  '/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/dybaux-5088.tar.gz'
	  File: `/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/dybaux-5088.tar.gz'
	  Size: 1915045602      Blocks: 3744000    IO Block: 4096   regular file
	Device: 306h/774d       Inode: 509413      Links: 1
	Access: (0644/-rw-r--r--)  Uid: (  506/dayabayscp)   Gid: (  506/dayabayscp)
	Access: 2011-10-21 20:03:12.000000000 +0800
	Modify: 2011-10-21 15:15:40.000000000 +0800
	Change: 2011-10-21 20:03:12.000000000 +0800
	[blyth@cms01 home]$ date
	Fri Oct 21 20:05:04 CST 2011

Appears that rsync has backdated the "Modify" timestamp to correspond to the source 
(this makes the reported transfer rate better than it actually was).


For the source tarball:

#. "Modify" = "Change"  corresponds to the last change (the file timestamp)
#. "Access" keeps updating as the rsync progresses

Reporting for the static source tarball is incorrect/misleading, other than size::

	[dayabay] /home/blyth/env > python ~/e/base/stat.py /home/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/dybaux-5088.tar.gz
	seconds 13024.0 delta 3:37:04       1915045602 bytes    1915.046 MB        0.147 MBps (during transfer) 

	[dayabay] /home/blyth/env > stat /home/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/dybaux-5088.tar.gz
	  File: `/home/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/dybaux-5088.tar.gz'
	  Size: 1915045602      Blocks: 3743992    IO Block: 4096   regular file
	Device: 6803h/26627d    Inode: 14193711    Links: 1
	Access: (0644/-rw-r--r--)  Uid: (    0/    root)   Gid: (    0/    root)
	Access: 2011-10-21 19:14:00.000000000 +0800
	Modify: 2011-10-21 15:15:40.000000000 +0800
	Change: 2011-10-21 15:15:40.000000000 +0800


Transfer time estimate for the 1.9GB tarball (at 0.102MBps)  ~5hrs 

	In [23]: from datetime import timedelta
	In [25]: est = timedelta(seconds=1915.046/0.102)
	In [26]: print est
	5:12:54.960784



The rate appears quite stable today::

	[blyth@cms01 home]$ ~/e/base/stat.py /data/var/scm/backup/dayabay/svn/dybsvn/2011/10/20/151703/.dybsvn-14745.tar.gz.8yxd9y
	seconds 1611.0 delta 0:26:51        175636480 bytes     175.636 MB        0.109 MBps (during transfer) 




"""
import os, stat
from datetime import datetime, timedelta

class Stat(str):
    mtime = property(lambda self:self._stat[stat.ST_MTIME])
    atime = property(lambda self:self._stat[stat.ST_ATIME])
    ctime = property(lambda self:self._stat[stat.ST_CTIME])

    mdate = property(lambda self:datetime.fromtimestamp(self.mtime))
    adate = property(lambda self:datetime.fromtimestamp(self.atime))
    cdate = property(lambda self:datetime.fromtimestamp(self.ctime))

    size = property(lambda self:self._stat[stat.ST_SIZE])
    MB   = property(lambda self:float(self.size)/1000000.)

    mode = property(lambda self:self._stat[stat.ST_MODE])
    isdir = property(lambda self:stat.S_ISDIR(self.mode))

    MBps = property(lambda self:self.MB/self.seconds)

    times = property(lambda self:(self.mtime, self.atime, self.ctime ))

    def _seconds(self):
        times = self.times 
        return float(max(times)-min(times))
    seconds = property(_seconds)
    
    delta  = property(lambda self:timedelta(seconds=self.seconds)) 

    def _age(self):
        td = datetime.now() - self.mdate
        return float( td.days * 3600 * 24 + td.seconds ) / ( 3600 * 24 )
    age  = property(_age )

    def __repr__(self):
        return "seconds %s delta %s  %15s bytes  %10.3f MB   %10.3f MBps (during transfer) " % ( self.seconds, self.delta, self.size, self.MB, self.MBps )

    def __init__(self, arg):
        super(self.__class__,self).__init__(arg)
        self._stat = os.stat(self)

if __name__ == '__main__':
    import sys
    s = Stat(sys.argv[1])
    print repr(s)


 



