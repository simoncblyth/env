#!/usr/bin/env python
"""
   Extracted from 
     trac/db/sqlite_backend.py

"""
try:
    import pysqlite2.dbapi2 as sqlite
    have_pysqlite = 2
except ImportError:
    try:
        import sqlite3 as sqlite
        have_pysqlite = 2
    except ImportError:
        try:
            import sqlite
            have_pysqlite = 1
        except ImportError:
            have_pysqlite = 0

if have_pysqlite == 2:
    _ver = sqlite.sqlite_version_info
    sqlite_version = _ver[0] * 10000 + _ver[1] * 100 + int(_ver[2])
    sqlite_version_string = '%d.%d.%d' % (_ver[0], _ver[1], int(_ver[2]))


elif have_pysqlite == 1:
    _ver = sqlite._sqlite.sqlite_version_info()
    sqlite_version = _ver[0] * 10000 + _ver[1] * 100 + _ver[2]
    sqlite_version_string = '%d.%d.%d' % _ver

else:
    sqlite_version_string = "none"


print "sqlite_version_string:%s have_pysqlite:%s" % ( sqlite_version_string , have_pysqlite )

