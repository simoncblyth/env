#!/usr/bin/env python

from bsddb3.db import *
from dbxml import *
v = version()
assert v == (4, 8, 26), "output must be the version of DB bundled with BDB XML %s" % repr(v) 

mgr=XmlManager()
print mgr.get_version_string()

