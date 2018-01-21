#!/usr/bin/env python
"""
This is separate to allow usage from both high (trac2sphinx.py) 
and low (test_tracwiki2rst.py) levels.
"""
import os, logging
log = logging.getLogger(__name__)


class Resolver(dict):
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)

    def getpath(self, name, ext=".rst"):
        path = os.path.join(self["rstdir"], "%s%s" % (name,ext) )
        dir_ = os.path.dirname(path)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        pass
        return path


if __name__ == '__main__':
    pass


