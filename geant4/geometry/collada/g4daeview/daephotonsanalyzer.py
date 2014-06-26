#!/usr/bin/env python
"""
Usage from ipython::

   In [224]: run daephotonsanalyzer.py
   In [226]: z = analyzer 
   In [225]: a = z.propagated


Questions of photon propagation histories 

#. do many step histories have something different about them,  
 
   * wavelength ? REEMISSION 


"""
import logging, os
import numpy as np
log = logging.getLogger(__name__)
from photons import mask2arg_, count_unique

class DAEPhotonsAnalyzer(object):
    """
    Interpret information recorded during and at tail 
    of propagate_vbo.cu:propagate_vbo
    """
    path = "propagated.npz"
    def __init__(self, max_slots, slot=-1 ):
        self.max_slots = max_slots
        self.slot = slot
        self.propagated = None

    def load(self, path=None):
        path = self.path if path is None else path
        log.info("load propagated from %s " % path )
        path = os.path.expanduser(os.path.expandvars(path))
        with np.load(path) as npz:
            propagated = npz['propagated']
        pass
        self(propagated)

    def write_propagated(self, path=None):
        assert not self.propagated is None
        path = self.path if path is None else path
        log.info("write propagated into %s " % path )
        np.savez_compressed(path, propagated=self.propagated)

    def __call__(self, propagated):
        """
        :param propagated: 
        """
        log.debug("analyzer.__call__")
        if propagated is None:return
        self.propagated = propagated
        self.analyze() 

    def get_vector(self, field='last_hit_triangle', index=0):
        """
        #. devious indexing to get top slot by viewing backwards
        """ 
        max_slots = self.max_slots
        slot = self.slot
        if slot == -1:
            vec = self.propagated[field][::-max_slots,index][::-1]
        elif slot == 0:
            vec = self.propagated[field][::max_slots,index]
        else:
            assert 0, slot 
        pass
        return vec

    ## accessors

    flags = property(lambda self:self.get_vector(field='flags', index=0))
    t0    = property(lambda self:self.get_vector(field='flags', index=1).view(np.float32))
    tf    = property(lambda self:self.get_vector(field='flags', index=2).view(np.float32))
    time_range = property(lambda self:[0.,self.tf.max()])  # start from 0, not min

    lht       = property(lambda self:self.get_vector(field='last_hit_triangle', index=0))
    photon_id = property(lambda self:self.get_vector(field='last_hit_triangle', index=1)) 
    steps     = property(lambda self:self.get_vector(field='last_hit_triangle', index=2))
    slots     = property(lambda self:self.get_vector(field='last_hit_triangle', index=3))

    history = property(lambda self:count_unique(self.flags))
    hsteps  = property(lambda self:count_unique(self.steps))
    hslots  = property(lambda self:count_unique(self.slots))


    def _get_counts_firsts_drawcount(self):
        """Counts with truncation, indices of start of each photon record"""
        photon_id = self.photon_id
        nphoton = len(photon_id)
        counts = np.clip( self.slots, 0, self.max_slots-2 ) 
        firsts = np.arange(nphoton, dtype='i')*self.max_slots
        drawcount = nphoton
        return counts, firsts, drawcount
    counts_firsts_drawcount = property(_get_counts_firsts_drawcount, doc=_get_counts_firsts_drawcount.__doc__)

    ## steering 

    def analyze(self):
        propagated = self.propagated
        if propagated is None:return
        
        self.dump()
        self.check_history()
        self.check_flags()
        self.check_steps()
        self.check_counts_firsts_drawcount()

    ## checks 

    def dump(self):
        log.info("dump")
        propagated = self.propagated
        print propagated
        print propagated.dtype
        print propagated.size
        print propagated.itemsize

    def check_history(self):
        log.info("check_history")
        history = self.history 
        present_history_ = lambda _:"%5s %-80s %s " % ( _[0], mask2arg_(_[0]), _[1] )
        print "\n".join(map(present_history_,history))

    def check_flags(self):
        log.info("check_flags")
        flags = self.flags
        t0 = self.t0
        tf = self.tf
        time_range = self.time_range
 
        print "flags", flags
        print "t0", t0
        print "tf", tf
        print "t0 range ", t0.min(), t0.max()
        print "tf range ", tf.min(), tf.max()
        print "time_range ", time_range

    def check_steps(self):
        """
        Looking at how many high-stepping histories there are, 
        or how much of a problem are truncations.
        """
        log.info("check_steps")
        hsteps = self.hsteps 
        present_hsteps_ = lambda _:"%5s %s " % ( _[0], _[1] )
        print "\n".join(map(present_hsteps_,hsteps))

    def check_counts_firsts_drawcount(self):
        log.info("check_counts_firsts_drawcount")
        photon_id = self.photon_id
        assert np.all(np.arange(0,len(photon_id),dtype=np.int32) == photon_id)

        lht = self.lht
        steps = self.steps
        slots = self.slots    
 
        #assert np.all( lht == -1 )  #no longer the case, as are now putting last slot result into slot 0
        print "lht", lht 
        if not np.all( steps == slots ):
            log.debug("steps and slots differ\nsteps:%s\nslots:%s" % (repr(steps),repr(slots)))

        counts, firsts, drawcount = self.counts_firsts_drawcount
        assert len(counts) == len(firsts) == len(photon_id)

        log.debug( " counts %s " % str(counts))
        log.debug( " firsts %s " % str(firsts))
        log.debug( " drawcount %s " % str(drawcount))



if __name__ == '__main__':
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()

    analyzer = DAEPhotonsAnalyzer( max_slots=config.args.max_slots )
    analyzer.load("~/e/propagated.npz")


 
