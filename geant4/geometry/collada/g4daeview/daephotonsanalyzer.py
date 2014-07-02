#!/usr/bin/env python
"""
From commandline::

    delta:~ blyth$ cd /usr/local/env/tmp/20140514-175600/    ## get path for currently loaded event from g4daeview stdout 
    delta:20140514-175600 blyth$ daephotonsanalyzer.sh propagated-0.npz 

#. hmm, it would be useful for g4daeview.py to vend some JSON regarding current high level state

Handling Truncation 
--------------------

#. reserve slot -2, and use it for the last propagated position (up to propagation max_steps)

   * motivations eg to be able to select a photon, need to have easy way to access final position

#. animation interpolation presentation can continue to use slot -1  

::

    b = a.reshape( (4165,10) )


"""
import logging, os, filecmp
import numpy as np
log = logging.getLogger(__name__)
from photons import mask2arg_, count_unique
from daephotonscompare import DAEPhotonsCompare


def srep(obj, att, index):
    """
    :param obj:
    :param att: attribute name
    :param index: index of array obj.<att>[index]

    :return: string representation of numpy array with att[index] label
    """
    body = str(getattr(obj,att)[index])  
    maxl = max(map(len,body.split("\n"))) 
    label = "%s[%s]" % (att,index)
    fmt = "%-"+str(maxl)+"s"
    fmt_ = lambda _:fmt % _
    ubody = "\n".join(map(fmt_,body.split("\n")))
    return "\n".join([fmt_(label),ubody])

def side_by_side(*arr):
    """
    :param arr: list of string representations of numpy arrays, 
                all the representations need to have the same length

    :return:  sting with the arrays presented side by side
    """
    split = map(lambda _:_.split("\n"),arr)
    zsplit = zip(*split)
    return "\n".join(map(lambda _:" ".join(_), zsplit))

def att_side_by_side( obj, index, atts):
    """
    :param obj:
    :param index:
    :param atts: list of attribute names
    """
    srep_ = lambda _:srep(obj,_,index)
    return side_by_side(*map(srep_,atts)) 




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

    # its always filecmp mismatching so no point checking
    #fcmp = filecmp.cmp(apath,bpath)
    #if not fcmp:
    #    log.warn("filecmp sees binary MISMATCH between %s %s but %s np mismatches  " % (apath, bpath, mismatch))
    #else:
    #    log.info("filecmp sees match between %s %s np mismatch %s " % (apath, bpath, mismatch))
    pass
    return mismatch 


def nearest_index(a,a0):
    """
    http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    return np.sum(np.square(np.abs(a-a0)),1).argmin()


class DAEPhotonsPropagated(object):
    def __init__(self, propagated=None, max_slots=10, slot=-1 ):
        self.max_slots = max_slots
        self.slot = slot
        if not propagated is None:
            self(propagated)

    def __call__(self, propagated):
        self.propagated = propagated

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

    # max_slots chunked arrays, for per-photon rather than per-step view 
    p_post = property(lambda self:self.propagated['position_time'].reshape(-1,self.max_slots,4))
    p_dirw = property(lambda self:self.propagated['direction_wavelength'].reshape(-1,self.max_slots,4))
    p_polw = property(lambda self:self.propagated['polarization_weight'].reshape(-1,self.max_slots,4))
    p_ccol = property(lambda self:self.propagated['ccolor'].reshape(-1,self.max_slots,4))
    p_flags = property(lambda self:self.propagated['flags'].reshape(-1,self.max_slots,4))
    p_lht   = property(lambda self:self.propagated['last_hit_triangle'].reshape(-1,self.max_slots,4))

    nphoton = property(lambda self:len(self.propagated)/self.max_slots)

    def _get_indices(self, slot=-1):
        return np.arange( self.nphoton )*self.max_slots + (self.max_slots+slot)
    last_index = property(lambda self:self._get_indices(slot=-2))

    ## slot -2 accessors 
    last_post  = property(lambda self:self.propagated['position_time'][self.last_index])
    last_dirw  = property(lambda self:self.propagated['direction_wavelength'][self.last_index])
    last_polw  = property(lambda self:self.propagated['polarization_weight'][self.last_index])
    last_ccol  = property(lambda self:self.propagated['ccolor'][self.last_index])
    last_flags = property(lambda self:self.propagated['flags'][self.last_index])
    last_lht   = property(lambda self:self.propagated['last_hit_triangle'][self.last_index])

    def nearest_photon(self, click):
        """
        :return: index of photon with final resting place closest to world frame coordinate `click` 
        """
        last_post = self.last_post
        index = nearest_index( last_post[:,:3], click)
        delta = click - last_post[:,:3][index]
        log.info("nearest_photon to click %s index %s at %s delta %s " % ( repr(click), index, last_post[index], repr(delta)  )) 
        return index

    ## slot -1 accessors 
    t_post  = property(lambda self:self.propagated['position_time'][::-self.max_slots][::-1])
    t_dirw  = property(lambda self:self.propagated['direction_wavelength'][::-self.max_slots][::-1])
    t_polw  = property(lambda self:self.propagated['polarization_weight'][::-self.max_slots][::-1])
    t_ccol  = property(lambda self:self.propagated['ccolor'][::-self.max_slots][::-1])
    t_flags = property(lambda self:self.propagated['flags'][::-self.max_slots][::-1])
    t_lht   = property(lambda self:self.propagated['last_hit_triangle'][::-self.max_slots][::-1])

    def t_nearest_photon(self, click):
        """
        :return: index of photon with time dependent position closest to world frame coordinate `click` 

        NB for this to return correct indices the VBO needs to have been pulled off the GPU 
        very recently 
        """ 
        t_post = self.t_post
        index = nearest_index( t_post[:,:3], click)
        delta = click - t_post[:,:3][index]
        log.info("t_nearest_photon to click %s index %s at %s delta %s " % ( repr(click), index, t_post[index], repr(delta)  )) 
        return index

    def summary(self, pid):
        log.info("summary for pid %s " % pid )
        #atts = "t_post".split()
        #for att in atts:
        #    print att
        #    print getattr(self,att)[pid]
        #pass
        print att_side_by_side(self, pid, "p_flags p_lht".split()) 
        print att_side_by_side(self, pid, "p_post p_dirw p_polw p_ccol".split()) 
        print att_side_by_side(self, pid, "t_post t_dirw t_polw t_ccol".split()) 

    def _get_counts_firsts_drawcount(self):
        """Counts with truncation, indices of start of each photon record"""
        photon_id = self.photon_id
        nphoton = len(photon_id)
        counts = np.clip( self.slots, 0, self.max_slots-2 )  ## does this need to change with new slot -2 scheme ?
        firsts = np.arange(nphoton, dtype='i')*self.max_slots
        drawcount = nphoton
        return counts, firsts, drawcount

    counts_firsts_drawcount = property(_get_counts_firsts_drawcount, doc=_get_counts_firsts_drawcount.__doc__)




class DAEPhotonsAnalyzer(DAEPhotonsPropagated):
    """
    Interpret information recorded during and at tail 
    of propagate_vbo.cu:propagate_vbo
    """
    name = "propagated-%(seed)s.npz"
    def __init__(self, max_slots, slot=-1 ):
        DAEPhotonsPropagated.__init__(self, None, max_slots, slot)
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

    def write_propagated(self, seed, eventpath, wipepropagate=False):
        """
        :param seed:
        :param eventpath: 
        :param wipepropagate:  set with --wipepropagate

        Trigger this with --debugpropagate

        Propagated VBO files containing numpy arrays are written 
        to a directory named after the basename of the originating event file.

        For example the event file `/usr/local/env/tmp/1.root` has 
        propagated files written to `/usr/local/env/tmp/1/propagated-0.npz` where
        the zero corresponds to the seed in use.

        When there is a preexisting output file they are 
        compared and an assertion is triggered if there is any mismatch 
        """
        assert not self.propagated is None
        if eventpath is None:
            log.warn("cannot write_propagated with event that has not been saved to file and subsequently loaded") 
            return
        pass
        name = self.name % locals() 
        eventbase, ext = os.path.splitext(eventpath)
        assert ext == '.root', ext 

        if not os.path.exists(eventbase):
            os.makedirs(eventbase)

        path = os.path.join( eventbase, name ) 
        log.info("write_propagated %s " % path ) 

        if wipepropagate and os.path.exists(path):
            log.info("removing invalidated prior %s due to --wipepropagate option   " % path )
            os.unlink(path)

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

        DAEPhotonsPropagated.__call__(self, propagated)

        self.analyze() 
    ## accessors
    atts = "propagated flags t0 t0 time_range lht photon_id steps slots history hsteps hslots".split()

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

    # populate context with some useful constants
    NO_HIT           = 0x1 << 0
    BULK_ABSORB      = 0x1 << 1
    SURFACE_DETECT   = 0x1 << 2
    SURFACE_ABSORB   = 0x1 << 3
    RAYLEIGH_SCATTER = 0x1 << 4
    REFLECT_DIFFUSE  = 0x1 << 5
    REFLECT_SPECULAR = 0x1 << 6
    SURFACE_REEMIT   = 0x1 << 7
    SURFACE_TRANSMIT = 0x1 << 8
    BULK_REEMIT      = 0x1 << 9
    NAN_ABORT        = 0x1 << 31

    STATUS_NONE = 0 
    STATUS_HISTORY_COMPLETE = 1
    STATUS_UNPACK = 2
    STATUS_NAN_FAIL = 3
    STATUS_FILL_STATE = 4 
    STATUS_NO_INTERSECTION = 5
    STATUS_TO_BOUNDARY = 6
    STATUS_AT_SURFACE = 7
    STATUS_AT_SURFACE_UNEXPECTED = 8
    STATUS_AT_BOUNDARY = 9
    STATUS_BREAKOUT = 10
    STATUS_ENQUEUE = 11
    STATUS_DONE = 12

    log.info("dropping into IPython.embed() try: z.<TAB> ")
    import IPython 
    IPython.embed()


if __name__ == '__main__':
    main()


 
