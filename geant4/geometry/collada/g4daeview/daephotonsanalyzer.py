#!/usr/bin/env python
"""
From commandline::

    delta:~ blyth$ daephotonsanalyzer.sh propagated-0.npz 


"""
import logging, os, filecmp
import numpy as np
log = logging.getLogger(__name__)
from photons import mask2arg_, count_unique
from daephotonscompare import DAEPhotonsCompare


def compare(apath, bpath, max_slots):
    """
    Compare persisted propagation npz files
    """ 
    a = DAEPhotonsAnalyzer.make(apath, max_slots)
    b = DAEPhotonsAnalyzer.make(bpath, max_slots)
    assert a.atts == b.atts
    cf = DAEPhotonsCompare( a, b )
    print cf
    mismatch = cf.compare(a.atts) 

    fcmp = filecmp.cmp(apath,bpath)
    if not fcmp:
        log.warn("filecmp sees binary MISMATCH between %s %s but %s np mismatches  " % (apath, bpath, mismatch))
    else:
        log.info("filecmp sees match between %s %s np mismatch %s " % (apath, bpath, mismatch))
    pass
    return mismatch 



class DAEPhotonsAnalyzer(object):
    """
    Interpret information recorded during and at tail 
    of propagate_vbo.cu:propagate_vbo
    """
    path = "propagated-%(seed)s.npz"
    def __init__(self, max_slots, slot=-1 ):
        self.max_slots = max_slots
        self.slot = slot
        self.propagated = None
        self.loaded = None

    @classmethod
    def make(cls, path, max_slots):
        analyzer = cls( max_slots=max_slots )
        analyzer.load(path)
        return analyzer

    def load(self, path=None):
        path = self.path if path is None else path
        log.info("load propagated from %s " % path )
        path = os.path.expanduser(os.path.expandvars(path))
        with np.load(path) as npz:
            propagated = npz['propagated']
        pass
        self.loaded = path
        self(propagated)

    def write_propagated(self, seed, path=None):
        """
        Trigger this with --debugpropagate

        When there is a preexisting output file they are 
        compared and an assertion is triggered if there is any mismatch 
        """
        assert not self.propagated is None
        path = self.path % locals() if path is None else path
        if not os.path.exists(path):
            self._write_propagated(path)
        else:
            tmppath = path.replace(".npz","-tmp.npz")
            self._write_propagated(tmppath)
            self.compare_propagated(path, tmppath)
        pass

    def _write_propagated(self, path):
        log.info("_write_propagated %s " % path )  
        np.savez_compressed(path, propagated=self.propagated)

    def compare_propagated(self, a, b):
        mismatch = compare(a, b, self.max_slots )
        assert mismatch == 0 , mismatch

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
    atts = "propagated flags t0 t0 time_range lht photon_id steps slots history hsteps hslots".split()

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

    def analyze(self, checks=False):
        propagated = self.propagated
        if propagated is None:return
        if not checks:return

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




def main():
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()
    clargs = config.args.clargs 
    assert len(clargs) > 0, "expecting commandline argument with path to npz file "
    path = clargs[0]

    log.info("creating DAEPhotonsAnalyzer for %s " % path )

    z = DAEPhotonsAnalyzer( max_slots=config.args.max_slots )
    z.load(path)

    log.info("dropping into IPython.embed() try: z.<TAB> ")
    import IPython 
    IPython.embed()


if __name__ == '__main__':
    main()


 
