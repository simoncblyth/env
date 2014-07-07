#!/usr/bin/env python
"""
See propagated_flags_mismatch.rst

"""

import logging, os
import numpy as np
log = logging.getLogger(__name__)


class DAEPhotonsCompare(object):
    def __init__(self, a, b ):
        """
        :param a: eg DAEPhotonsAnalyzer instance
        :param b: eg DAEPhotonsAnalyzer instance
        """ 
        self.a = a
        self.b = b

    def __str__(self):
        return "%s %s %s " % (self.__class__.__name__, self.a.loaded, self.b.loaded )

    def compare(self, atts): 
        mismatch = 0
        log.debug("comparing atts %s " % repr(atts))
        for att in atts:
            cf = np.all( getattr(self.a, att) == getattr(self.b, att))
            if not cf:
                log.warn("att %s cf %s " % ( att, cf ))
                mismatch += 1
                if att == 'propagated':
                    self.compare_propagated()

            else:
                log.debug(" att %s OK " % att )
        pass
        return mismatch

    def compare_propagated(self):
        assert self.a.propagated.dtype.names == self.b.propagated.dtype.names
        mismatch = 0 
        for name in self.a.propagated.dtype.names:
            cf = np.all( self.a.propagated[name] == self.b.propagated[name] ) 
            if not cf:
                log.warn( "compared_propagated %s %s" % (name, cf )) 
                mismatch += 1
            else: 
                log.debug( "compared_propagated %s %s" % (name, cf )) 
            pass
        return mismatch


def main():
    from daeconfig import DAEConfig
    from daephotonsanalyzer import DAEPhotonsAnalyzer

    config = DAEConfig()
    config.init_parse()

    paths = ("propagated-0.npz","propagated-0-tmp.npz",)

    a = DAEPhotonsAnalyzer.make(paths[0], config)
    b = DAEPhotonsAnalyzer.make(paths[1], config)
    assert a.atts == b.atts

    cf = DAEPhotonsCompare( a, b )
    print cf
    mismatch = cf.compare(a.atts) 
    if mismatch != 0:
        log.warn("ATTS MISMATCH %s " % mismatch )
   
    mismatch = cf.compare_propagated() 
    if mismatch != 0:
        log.warn("PROPAGATED MISMATCH %s " % mismatch )



    import IPython
    IPython.embed() 



if __name__ == '__main__':
    main()




    



