#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)

class Resolver(object):
    """
    """
    def __init__(self, args):
        ctx = {}
        ctx["tracdir"] = args.tracdir
        ctx["rstdir"] = args.rstdir
        self.ctx = ctx

    def __call__(self, ref, pagename):
        self.ctx['pagerel'] = ref
        self.ctx['pagename'] = pagename 
        return "file://%(tracdir)s/attachments/wiki/%(pagename)s/%(pagerel)s" % self.ctx
 
    def getpath(self, name, ext=".rst"):
        path = os.path.join(self.ctx["rstdir"], "%s%s" % (name,ext) )
        dir_ = os.path.dirname(path)
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        pass
        return path


if __name__ == '__main__':
    pass


