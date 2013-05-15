#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)

class CommandLine(dict):
    """ 
    Base class for composing and invoking command lines in a separate process 
    """
    def _exepath(self):
        for cmd in self._exenames:
            which = os.popen("which %s" % cmd).read().rstrip("\n")
            if os.path.exists(which):
                return which
        return None
    exepath = property( _exepath )
    cmd      = property( lambda self:self._cmd % self )
    cmd_nopw = property( lambda self:self._cmd % dict(self, password="***") )

    def __init__(self, *args, **kwa ):
        dict.__init__(self, *args, **kwa )
        exe = self.exepath
        assert exe, "cannot find executable %r check your PATH " % self._exenames
        self['exepath']=exe
        self['path']="/dev/null"
        self['argline']=""

    def __str__(self):
        return "%s %s " % (self.__class__.__name__, self.cmd_nopw )

    def __call__(self, **kwa):
        verbose = kwa.pop('verbose', False)
        self.update(kwa)
        if verbose:
            log.info(self)
        else:
            log.debug(self)
        return os.popen(self.cmd).read()    


if __name__ == '__main__':
    pass

