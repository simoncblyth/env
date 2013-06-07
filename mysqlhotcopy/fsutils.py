#!/usr/bin/env python
"""
"""
import os
from pwd import getpwnam, getpwuid
from grp import getgrnam, getgrgid

def _percent( num, den ): 
    try:
        ret = (num/den) * 100.
    except ZeroDivisionError:
        ret = 0
    return ret 

def _get_gid(name):
    """Returns a gid, given a group name."""
    if getgrnam is None or name is None:
        return None
    try:
        result = getgrnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None

def _get_uid(name):
    """Returns an uid, given a user name."""
    if getpwnam is None or name is None:
        return None
    try:
        result = getpwnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None

def _get_usergroup(path):
    st = os.stat(path)
    uid = st.st_uid
    gid = st.st_gid
    user = getpwuid(uid)[0]
    group = getgrgid(gid)[0]
    return user, group

def copychown(src, dst):
    """
    Copy user and group ownership from `src` to `dst` files 
    which must both exist.
    """  
    st = os.stat(src)
    os.chown(dst, st.st_uid, st.st_gid)


def chown(filename, user=None, group=None):
    """Change owner user and group of the given `filename`.

    `user` and `group` can either be the uid/gid or the user/group names, and in
    that case, they are converted to their respective uid/gid.
    """

    if user is None and group is None:
        raise ValueError("user and/or group must be set")

    _user = None
    _group = None

    # if is None, let's pass -1 to os.chown(), that means "don't change it"
    if user is None:
        _user = -1
    else:
        # user can either be an int (the uid) or a string (the system username),
        # in the latter case, we have to convert it to the uid
        try:
            _user = int(user)
        except ValueError:
            _user = _get_uid(user)

    # same goes for the group
    if group is None:
        _group = -1
    else:
        try:
            _group = int(group)
        except ValueError:
            _group = _get_gid(group)

    # at the end, call chown, that's actually going to change the user/group
    os.chown(filename, _user, _group)


import os, shutil

def copytree(src, dst, symlinks=False, ignore=None):
    """
    This is `shutil.copytree` from py27 in order to use 
    for py23 py24 py25 py26 without worry about varying capabilities
    of different versions. With ownership copying added

    Recursively copy a directory tree using copy2().

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    XXX Consider this example code rather than the ultimate tool.

    """
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    os.makedirs(dst)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore)
            else:
                # Will raise a SpecialFileError for unsupported file types
                shutil.copy2(srcname, dstname)
                copychown(srcname, dstname)   # owner/group copying 
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except shutil.Error, err:
            errors.extend(err.args[0])
        except shutil.EnvironmentError, why:
            errors.append((srcname, dstname, str(why)))
    try:
        shutil.copystat(src, dst)
        copychown(src, dst)          # owner/group copying 
    except OSError, why:
        if WindowsError is not None and isinstance(why, WindowsError):
            # Copying file access times may fail on Windows
            pass
        else:
            errors.extend((src, dst, str(why)))
    if errors:
        raise Error, errors



class DiskUsage(dict):
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

    def asdict(self, keys="percent_used percent_free gb_free gb_total".split()):
        """
        truncated summary dict, suitable for storage
        """
        s = {}
        for k in filter(lambda k:k in keys, self.keys()):
            v = "%6.2f" % self[k]
            s[k] = v.lstrip().strip()
        return s
    def __str__(self):
        return repr(self.asdict())
    def __repr__(self):
        return self.__class__.__name__ + self.tmpl % self



if __name__ == '__main__':
    du = DiskUsage()
    print du
    assert du['gb_free'] > 10 , du 



