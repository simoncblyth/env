#!/usr/bin/env python

import os


def ok_str(s):
    try:
        str(s)
        ok = True
    except UnicodeEncodeError as e:
        print e
        ok = False
    pass
    return ok


def encoding_check(txt, args):
    """
    http://pysqlite.readthedocs.io/en/latest/sqlite3.html
    https://docs.python.org/2/howto/unicode.html

    http://www.utf8-chartable.de/unicode-utf8-table.pl?start=128&number=128&names=-&utf8=0x

    """
    assert type(txt) is unicode

    lines = txt.split("\n")
    s_lines = map(lambda u:u.encode("utf-8"), lines)

    for i, u in enumerate(lines): 
        assert type(u) is unicode
        s = u.encode("utf-8")
        ok = ok_str(s)
        print "%3d : %s : %s " % ( i, " " if ok else "*", s)
    pass
    print "\n".join(map(str,s_lines))





rstpath = "%s%s" % ( os.path.splitext(os.path.abspath(__file__))[0], ".rst")
print rstpath

for _ in open(rstpath, "r").readlines():
    print _,
    print str(_),




