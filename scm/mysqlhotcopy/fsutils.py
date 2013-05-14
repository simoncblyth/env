#!/usr/bin/env python
"""
"""
import os

def _percent( num, den ): 
    try:
        ret = (num/den) * 100.
    except ZeroDivisionError:
        ret = 0
    return ret 

class disk_usage(dict):
    """
    http://docs.python.org/2/library/statvfs.html

    ::

        statvfs.F_BSIZE Preferred file system block size.
        statvfs.F_FRSIZE Fundamental file system block size.
        statvfs.F_BLOCKS Total number of blocks in the filesystem.
        statvfs.F_BFREE Total number of free blocks.
        statvfs.F_BAVAIL Free blocks available to non-super user.
        statvfs.F_FILES Total number of file nodes.
        statvfs.F_FFREE Total number of free file nodes.
        statvfs.F_FAVAIL Free nodes available to non-super user.

    Values are a bit different to those from `df` maybe rounding ? Or perhaps:

    * http://larsmichelsen.com/open-source/answer-why-does-df-k-show-wrong-percentage-usage/

    ::

        [blyth@belle7 DybPython]$ df -h
        Filesystem            Size  Used Avail Use% Mounted on
        ...                   865G  333G  487G  41% /

        [blyth@belle7 DybPython]$ ./fstools.py 
        disk_usage   free 486.83 G [56.33 %]    used 332.78 G [38.51 %]       total 864.21 G  


    """
    tmpl = "   free %(gb_free)-4.2f G [%(percent_free)-3.2f %%]    used %(gb_used)-4.2f G [%(percent_used)-3.2f %%]       total %(gb_total)-4.2f G  " 
    def __init__(self, path=None):
        dict.__init__(self)
        if not path:
            path = os.getcwd()
        if not os.path.exists(path):
            path = os.path.dirname(path)

        st = os.statvfs(path)
        self['bytes_free'] = float(st.f_bavail * st.f_frsize)
        self['bytes_total'] = float(st.f_blocks * st.f_frsize)
        self['bytes_used'] = float((st.f_blocks - st.f_bfree) * st.f_frsize)

        self['percent_used'] = _percent(self['bytes_used'], self['bytes_total'] )
        self['percent_free'] = _percent(self['bytes_free'], self['bytes_total'] )

        b2gb = 1./float(1024*1024*1024)
        self['gb_free'] = self['bytes_free']*b2gb
        self['gb_total'] = self['bytes_total']*b2gb
        self['gb_used'] = self['bytes_used']*b2gb

        b2mb = 1./float(1024*1024)
        self['mb_free'] = self['bytes_free']*b2mb
        self['mb_total'] = self['bytes_total']*b2mb
        self['mb_used'] = self['bytes_used']*b2mb


    def __repr__(self):
        return self.__class__.__name__ + self.tmpl % self


if __name__ == '__main__':
    du = disk_usage()
    print du
    assert du['gb_free'] > 10 , du 



