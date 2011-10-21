"""
Monitor file growth using stat, eg for checking rsync transfer of tarballs, usage::

	[blyth@cms01 base]$ python stat.py  '/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/.dybaux-5088.tar.gz.B3FMx6'
	delta 3:07:30       1149239296 bytes    1149.239 MB        0.102 MBps (during transfer) 

During a transfer observe that: 

#. "Access" corresponds to creation 
#. "Modify" = "Change"  corresponds to last update, usually slightly earlier than current time 


[blyth@cms01 ~]$ stat /data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/.dybaux-5088.tar.gz.B3FMx6
  File: `/data/var/scm/backup/dayabay/svn/dybaux/2011/10/21/151002/.dybaux-5088.tar.gz.B3FMx6'
  Size: 961806336       Blocks: 1880384    IO Block: 4096   regular file
Device: 306h/774d       Inode: 509413      Links: 1
Access: (0600/-rw-------)  Uid: (  506/dayabayscp)   Gid: (  506/dayabayscp)
Access: 2011-10-21 15:32:42.000000000 +0800
Modify: 2011-10-21 18:17:39.000000000 +0800
Change: 2011-10-21 18:17:39.000000000 +0800
[blyth@cms01 ~]$ 


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
    
    def _delta(self):
        return timedelta( seconds=self.seconds ) 
    delta  = property(_delta )

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


 



