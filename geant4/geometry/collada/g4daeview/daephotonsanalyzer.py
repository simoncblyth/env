#!/usr/bin/env python
"""
Usage::

    delta:~ blyth$ daephotonsanalyzer.sh --load 1


Changes
---------

#. metadata regarding the propagation now travels in sidecar .json files 


"""
import logging, os, filecmp, json, datetime
import numpy as np
log = logging.getLogger(__name__)
from photons import mask2arg_, count_unique
from daephotonscompare import DAEPhotonsCompare

timestamp = lambda:datetime.datetime.now()

def columnize( s ):
    """
    :param s: list of strings of varying lengths
    :return: list of strings formatted to all be the same length  
    """
    maxl = max(map(len, s))
    fmt_ = lambda _:("%-"+str(maxl)+"s") % _            
    return map(fmt_, s )


def srep(obj, att, index, transform_ = lambda _:_):
    """
    :param obj:
    :param att: attribute name
    :param index: index of array obj.<att>[index]

    :return: string representation of numpy array with att[index] label
    """
    body = str(transform_(getattr(obj,att)[index]))  
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



def att_side_by_side( obj, index, atts, tmap={}):
    """
    :param obj:
    :param index:
    :param atts: list of attribute names
    :param tmap: dict containing numpy array transform functions keyed by attribute name
    """
    identity_ = lambda _:_
    arr = []
    for att in atts:
        transform_ = tmap.get(att,identity_)
        arr.append(srep(obj,att,index,transform_))
    pass
    return side_by_side(*arr) 



def compare(apath, bpath):
    """
    Compare persisted propagation npz files
    """ 
    log.info("compare apath %s bpath %s " % (apath,bpath))
    a = DAEPhotonsAnalyzer.make(apath)
    b = DAEPhotonsAnalyzer.make(bpath)
    assert a.atts == b.atts
    assert a.max_slots == b.max_slots 
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
        self.slot = int(slot)
        self._last_index = None
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


    flags = property(lambda self:self.propagated['flags'][self.last_index][:,0])
    t0    = property(lambda self:self.propagated['flags'][self.last_index][:,1].view(np.float32))
    tf    = property(lambda self:self.propagated['flags'][self.last_index][:,2].view(np.float32))
    tl    = property(lambda self:self.tf - self.t0)
    steps = property(lambda self:self.propagated['flags'][self.last_index][:,3])
    time_range = property(lambda self:[0.,self.tf.max()])  # start from 0, not min

    lht   = property(lambda self:self.propagated['last_hit_triangle'][self.last_index][:,0]) 
    mat1  = property(lambda self:self.propagated['last_hit_triangle'][self.last_index][:,1]) 
    mat2  = property(lambda self:self.propagated['last_hit_triangle'][self.last_index][:,2]) 
    slots = property(lambda self:self.propagated['last_hit_triangle'][self.last_index][:,3]) 

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
    p_i     = property(lambda self:np.tile(np.arange(self.max_slots), self.nphoton).reshape(-1,self.max_slots,1))

    long_lived = property(lambda self:np.where( self.tl > 200. )[0])  # lifetime more than 200 ns, look a lot better with high max_slots 
    special = property(lambda self:self.long_lived)

    nphoton = property(lambda self:len(self.propagated)/self.max_slots)

    def _get_indices(self, slot=-1):
        return np.arange( self.nphoton )*self.max_slots + (self.max_slots+slot)
    #last_index = property(lambda self:self._get_indices(slot=-2))

    def _get_last_index(self):
        if self._last_index is None:
            log.info("_last_index")
            self._last_index = self._get_indices(slot=-2)
        return self._last_index
    last_index = property(_get_last_index)

    ## slot -2 accessors 
    last_post  = property(lambda self:self.propagated['position_time'][self.last_index])
    last_dirw  = property(lambda self:self.propagated['direction_wavelength'][self.last_index])
    last_polw  = property(lambda self:self.propagated['polarization_weight'][self.last_index])
    last_ccol  = property(lambda self:self.propagated['ccolor'][self.last_index])
    last_flags = property(lambda self:self.propagated['flags'][self.last_index])
    last_lht   = property(lambda self:self.propagated['last_hit_triangle'][self.last_index])

    # raw, all slot accessors
    material1  = property(lambda self:self.propagated['last_hit_triangle'][:,1])
    material2  = property(lambda self:self.propagated['last_hit_triangle'][:,2])
    matpair    = property(lambda self:self.material1*1000 + self.material2)   # converted into string in DAEChromaMaterialMap.paircode2str


    def find_material_pairs(self, mp, material_map):
        """
        :param mp: string like "Acrylic,GdDopedLS"
        """
        pc = material_map.str2paircode(mp)
        return np.where( self.matpair == pc )


    def nearest_photon(self, click):
        """
        :return: index of photon with final resting place closest to world frame coordinate `click` 
        """
        last_post = self.last_post
        index = nearest_index( last_post[:,:3], click)
        delta = click - last_post[:,:3][index]
        log.info("nearest_photon to click %s index %s at %s delta %s " % ( repr(click), index, last_post[index], repr(delta)  )) 
        return index

    def _get_is_enabled(self):
        if not hasattr(self, 'propagated'):return False
        if self.propagated is None:return False
        return True
    is_enabled = property(_get_is_enabled)

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

    def _get_counts_firsts_drawcount(self):
        """
        Counts with truncation, indices of start of each photon record

        np.clip restricts values gt max to be max and lt min to be min
        ie with max_slots = 10 slots gt max_slots-2 = 8 become 8 
      
        Suspect the counts may be one less than they should be, its the 
        slot value (which is zero based). Hence the + 1

        Clipping not actually required, but doing it makes the counts into
        a contiguous array, rather than sliced. This is needed by pyopengl
        """
        nphoton = self.nphoton
        counts = np.clip( self.slots + 1, 0, self.max_slots-2 )  
        assert np.all( counts == self.slots + 1 )
        firsts = np.arange(nphoton, dtype='i')*self.max_slots
        drawcount = nphoton
        return counts, firsts, drawcount

    counts_firsts_drawcount = property(_get_counts_firsts_drawcount, doc=_get_counts_firsts_drawcount.__doc__)

    def _get_stepindices(self):
        counts,firsts,drawcount = self.counts_firsts_drawcount
        ranges = np.vstack( [firsts, firsts+counts]).T
        return np.concatenate(map(lambda _:np.arange(*_), ranges))   # the map will be python slow 
    stepindices = property(_get_stepindices)

    

    def summary(self, pid, material_map=None, process_map=None):
        log.info("summary for pid %s " % pid )
        if material_map is None:
            material_map = self.material_map
        if process_map is None:
            process_map = self.process_map

        def format_p_flags(a):
            """
            fill_meta body::

                qflags.u.x = p.history ;
                qflags.f.y = s.distance_to_boundary ;
                qflags.f.z = 0. ;
                qflags.f.w = 0. ;

            last (-2)::

                qflags.u.x = p.history ;
                qflags.f.y = t0 ;             
                qflags.f.z = p.time ;   
                qflags.u.w = steps ; 


            """
            history = columnize(map( process_map.mask2str, a[:,0] ))
            b = np.empty((a.shape[0],2),dtype=np.float32)
            b[:,0] = a[:,1].view(np.float32)
            b[:,1] = a[:,2].view(np.float32)
            steps = str(a[:,3:])
            sbs = side_by_side( "\n".join(history),str(b),steps ) 
            return sbs

        def format_p_lht(a):
            from_material = columnize(map( material_map.code2str, a[:,1])) 
            to_material   = columnize(map( material_map.code2str, a[:,2])) 
            return side_by_side( str(a), "\n".join(from_material), "\n".join(to_material) )

        def format_p_post(a, maxdist=10000):
            """
            distance between step positions
            """
            dists = a[1:,0:3] - a[:-1,0:3]
            stepdist = [-1] + map(np.linalg.norm, dists)
            stepdist = map( lambda _:-1 if _>maxdist else _, stepdist)
            return side_by_side( str(a), "\n".join(map(str,stepdist)))

        tmap = {} 
        tmap['p_flags'] = format_p_flags
        tmap['p_lht'] = format_p_lht
        tmap['p_post'] = format_p_post

        print att_side_by_side(self, pid, "p_i p_flags p_lht".split(), tmap ) 
        print att_side_by_side(self, pid, "p_i p_post p_dirw p_polw p_ccol".split(), tmap ) 
        print att_side_by_side(self, pid, "t_post t_dirw t_polw t_ccol".split()) 







class DAEPhotonsAnalyzer(DAEPhotonsPropagated):
    """
    Interpret information recorded during and at tail 
    of propagate_vbo.cu:propagate_vbo
    """
    name = "propagated-%(seed)s.npz"
    def __init__(self, max_slots=None, slot=-1, material_map=None, process_map=None):
        DAEPhotonsPropagated.__init__(self, None, max_slots, slot)
        self.loaded = None
        self.material_map = material_map
        self.process_map = process_map

    @classmethod
    def make(cls, path ):
        analyzer = cls()
        analyzer.load(path)
        return analyzer

    def load(self, path=None):
        path = self.path if path is None else path
        log.info("load propagated from %s " % path )
        path = os.path.expanduser(os.path.expandvars(path))
        with np.load(path) as npz:
            propagated = npz['propagated']
        pass
        metadata = self._load_metadata( self.sidecar_path(path) )
        log.info("load metadata gives %s " % repr(metadata) )

        assert 'max_slots' in metadata
        self.max_slots = int(metadata['max_slots'])

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

        metadata = self.make_metadata() 
        if not os.path.exists(path):
            self._write_propagated(path, **metadata)
        else:
            tmppath = path.replace(".npz","-tmp.npz")
            self._write_propagated(tmppath, **metadata )
            self.compare_propagated(path, tmppath)
        pass

    def make_metadata(self):
        metadata = {}
        metadata['max_slots'] = self.max_slots
        now = timestamp()
        metadata['timestamp'] = now.strftime("%s")
        metadata['date'] = now.strftime("%Y%m%d-%H%M")
        return metadata

    def sidecar_path(self, npz ):
        return npz.replace(".npz",".json")

    def _write_propagated(self, path, **metadata):
        log.info("_write_propagated %s " % path )  
        np.savez_compressed(path, propagated=self.propagated)
        self._write_metadata( self.sidecar_path(path), **metadata )

    def _write_metadata(self, path, **metadata):
        if len(metadata) == 0:return
        log.info("writing to %s " % path )
        with open(path,"w") as fp:
            json.dump(metadata, fp) 

    def _load_metadata(self, path):
        """ 
        json keys and values are unicode strings by default, 
        so convert to str,str on reading to match the original dict 
        """
        if not os.path.exists(path):
            log.warn("no such path %s " % path)
            return None
        log.info("reading from %s " % path )
        with open(path,"r") as fp:
            pd = json.load(fp)
        pass
        return dict(map(lambda _:(str(_[0]),str(_[1])),pd.items()))

    def compare_propagated(self, a, b):
        mismatch = compare(a, b )
        if mismatch > 0:
            log.warn("compare_propagated mismatch %s " % mismatch )
            log.warn("Debug with eg: cd /usr/local/env/tmp/1/ ; daephotonscompare.sh --loglevel debug ")
        pass

    def get_material_pairs(self, material_map):
        items = []
        items.append(("ANY,ANY","ANY,ANY",))
        if not hasattr(self, 'propagated'):
            log.info("get_material_pairs needs propagated attribute")
            return items 
        if self.propagated is None:
            log.info("get_material_pairs needs propagated event")
            return items 
        pass
        vals = self.matpair
        mp = count_unique(vals)
        mps = mp[mp[:,-1].argsort()[::-1]]     # order by decreasing pair count  
        for mm,count in mps: 
            matcode = material_map.paircode2str(mm)
            matname = "%-4d %s " % ( count, matcode )
            items.append((matname,matcode))
        pass
        return items


    def __call__(self, propagated):
        """
        :param propagated: 
        """
        log.debug("analyzer.__call__")
        if propagated is None:return

        DAEPhotonsPropagated.__call__(self, propagated)

        self.analyze() 
    ## accessors
    atts = "propagated flags t0 t0 time_range lht steps slots history hsteps hslots".split()

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

        lht = self.lht
        steps = self.steps
        slots = self.slots    
 
        print "lht", lht 
        if not np.all( steps == slots ):
            log.debug("steps and slots differ\nsteps:%s\nslots:%s" % (repr(steps),repr(slots)))

        counts, firsts, drawcount = self.counts_firsts_drawcount
        assert len(counts) == len(firsts) 

        log.debug( " counts %s " % str(counts))
        log.debug( " firsts %s " % str(firsts))
        log.debug( " drawcount %s " % str(drawcount))

    def plot_steps(self):
        """
        Very long tail with a blip at 100 corresponidng to Chroma propagation truncation, 

        #. max_slots 10 introduces too much truncation distortion.  
        #. max_slots 30 probably good enough, 

        """
        import matplotlib.pyplot as plt
        plt.hist(self.steps, bins=25, label="steps") 
        plt.legend(title="Chroma propagation steps")
        plt.show()

    def plot_slots(self):
        """
        A third of the >50 are in the chroma truncation top bin 97
        """
        import matplotlib.pyplot as plt
        z = self
        h = plt.hist(z.slots[z.slots>50], bins=np.linspace(50,100,51))
        plt.show()

    def present_material_pairs(self):
        mps = self.get_material_pairs( self.material_map )
        print "\n".join( ["%-40s" % (mp[0]) for mp in mps])  

    def check_surface(self):
        """
        Huh, only 2 sufaces and one of them only once.

        ::

            In [11]: count_unique(surface)
            Out[11]: 
            array([[   -1, 96830],
                   [    1,  7160],
                   [   23,     1]])

        ::

            In [9]: np.intersect1d( np.where( asurface == 23 )[0], z.stepindices )
            Out[9]: array([312600])

            In [10]: z.summary(3126)   ## WaterPool PMT ?  
            2014-07-08 20:22:40,059 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:257 summary for pid 3126 
            p_flags[3126]                            p_lht[3126]                                              
                               [[  0.       0.    ]  [[2382597      27      10      23] OwsWater   UnstStainl    #######
            R_DIFFUSE           [  0.       0.    ]   [2165175       7       9      -1] Pyrex      Vacuum     
            B_ABSORB,R_DIFFUSE  [  0.       0.    ]   [     -1       7       9      -1] Pyrex      Vacuum     
            B_ABSORB,R_DIFFUSE  [ 21.7334  34.4839]   [      0       0       0       2] LiquidScin LiquidScin 

            # udp.py --style confetti,spagetti,noodles --sid 3129 
            # UnstStainlessSteel   surface 23 ?


            p_flags[1]                   p_lht[1]                                          
                     [[ 0.      0.    ]  [[621958     15      5      1] MineralOil Acrylic 
            B_ABSORB  [ 0.      0.    ]   [    -1     15      5      1] MineralOil Acrylic 


            In [11]: z.summary(4160)
            2014-07-08 21:00:38,346 env.geant4.geometry.collada.g4daeview.daephotonsanalyzer:257 summary for pid 4160 
            p_flags[4160]                                                  p_lht[4160]                                          
                                                   [[   0.        0.    ]  [[  1216      3      5     -1] GdDopedLS  Acrylic    
            R_SCATTER                               [   0.        0.    ]   [  1216      3      5     -1] GdDopedLS  Acrylic    
            R_SCATTER                               [   0.        0.    ]   [   928      5      4     -1] Acrylic    LiquidScin 
            R_SCATTER                               [   0.        0.    ]   [   598      4      5     -1] LiquidScin Acrylic    
            R_SCATTER                               [   0.        0.    ]   [   310      5     15     -1] Acrylic    MineralOil 
            R_SCATTER                               [   0.        0.    ]   [618757     15      5      1] MineralOil Acrylic   ######### 
            R_SCATTER,R_SPECULAR                    [   0.        0.    ]   [   310     15      5     -1] MineralOil Acrylic    
            R_SCATTER,R_SPECULAR                    [   0.        0.    ]   [   598      5      4     -1] Acrylic    LiquidScin 
            R_SCATTER,R_SPECULAR                    [   0.        0.    ]   [   931      4      5     -1] LiquidScin Acrylic    
            R_SCATTER,R_SPECULAR                    [   0.        0.    ]   [  1219      5      3     -1] Acrylic    GdDopedLS  

  
        ::

            In [4]: np.intersect1d( np.where( asurface == 1 )[0], z.stepindices )
            Out[4]: array([   100,    200,    300, ..., 416005, 416014, 416205])

            In [5]: np.intersect1d( np.where( asurface == 1 )[0], z.stepindices )//100
            Out[5]: array([   1,    2,    3, ..., 4160, 4160, 4162])


        daegeometry.sh::

            In [8]: cg.unique_surfaces[23]   #?
            Out[8]: <Surface __dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib4Surface>

            In [9]: cg.unique_surfaces[1]
            Out[9]: <Surface __dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface>


        """
        z = self
        asurface = z.propagated['last_hit_triangle'][:,-1]
        surface = asurface[z.stepindices]

        


def main():
    from daeconfig import DAEConfig
    from daechromaprocessmap import DAEChromaProcessMap
    from daechromamaterialmap import DAEChromaMaterialMap

    config = DAEConfig()
    config.init_parse()
    cpm = DAEChromaProcessMap.fromjson(config)
    cmm = DAEChromaMaterialMap.fromjson(config)
    #print cpm
    #print cmm

    clargs = config.args.clargs 
    if len(clargs) > 0:
        path = clargs[0]
    else:
        name = DAEPhotonsAnalyzer.name % dict(seed=config.args.seed)
        path = config.resolve_event_path( config.args.load , name )
        log.info("resolve %s %s => %s " % (config.args.load, name, path))
    pass
    log.info("creating DAEPhotonsAnalyzer for %s " % (path ))


    z = DAEPhotonsAnalyzer( max_slots=config.args.max_slots, material_map=cmm, process_map=cpm )
    z.load(path)
    z.summary(0)

    z.present_material_pairs()

    import matplotlib.pyplot as plt


    log.info("dropping into IPython.embed() try: z.<TAB> ")
    import IPython 
    IPython.embed()


if __name__ == '__main__':
    main()


 
