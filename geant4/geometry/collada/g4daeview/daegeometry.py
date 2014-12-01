#!/usr/bin/env python

import os, logging, re, time, traceback
log = logging.getLogger(__name__)
import numpy as np
from env.geant4.geometry.collada.g4daenode import DAENode 
from env.base.timing import timing, timing_report

from daeutil import printoptions, ModelToWorld, WorldToModel
from daeviewpoint import DAEViewpoint
from daechromamaterialmap import DAEChromaMaterialMap
from daechromasurfacemap import DAEChromaSurfaceMap
from daechromaprocessmap import DAEChromaProcessMap

from collada.xmlutil import etree as ET

tostring_ = lambda _:ET.tostring(getattr(_,'xmlnode')) 
shortname_ = lambda _:_[17:-9]   # trim __dd__Materials__GdDopedLS_fx_0xc2a8ed0 into GdDopedLS 


class Regexp(object):
    def __init__(self, ptn):
        """
        :param ptn: string search regexp 
        """
        self.ptn = re.compile(ptn)

    def select(self, nodes):
        """
        :param nodes: list of instances (eg DAENode or DAESolid) with .id attribute

        Usage example::

            daegeometry.sh --regexp PmtHemiCathode 

        """
        match_ = lambda _:_ if self.ptn.search(_.id) else None
        return filter(None,map(match_, nodes))


       




class DAEMesh(object):
    """
    TODO: remove use of duplicating pair properties, 
          now that caching in place these is no need for them

    Note that most of the useful features of DAEMesh only 
    requires a numpy array of vertices, so a ChromaPhotonList 
    can be treated as a DAEMesh for navigation purposes.
    """
    secs = {}
    @timing(secs)
    def __init__(self, vertices, triangles=[], normals=[] ):
        self._vertices = vertices
        self._triangles = triangles
        self._normals = normals

        self._index = -1
        self.id = "entiremesh"
        self.reset() 

    def reset(self):
        """
        Reset locally cached calculated values
        """ 
        self._bounds = None
        self._lower_upper = None
        self._center = None
        self._extent = None
        self._dimensions = None
        self._bounds_extent = None
        self._center_extent = None

    vertices = property(lambda self:self._vertices)
    triangles = property(lambda self:self._triangles)
    normals = property(lambda self:self._normals)

    def _get_index(self):
        return self._index 
    def _set_index(self, index):
        self._index = index
    index = property(_get_index,_set_index)

    def check(self):
        """
        Twelve solids fail this, all with same values::

             DAESolid Meshcheck failure DAESolid vertex 267  triangles 528  normals 267   : 4522 __dd__Geometry__CalibrationSources__lvWallLedSourceAssy--pvWallLedDiffuserBall0xab71f78.0 
        """
        assert np.min(self.triangles) == 0
        if np.max(self.triangles) != len(self.vertices)-1:
            return False
        return True

    def _get_bounds(self):
        "Return the lower and upper bounds for the mesh as a tuple."
        if self._bounds is None:
            self._bounds = np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
        return self._bounds
    bounds = property(_get_bounds)

    def _get_lower_upper(self):
        "Return the lower and upper bounds for the mesh as a tuple."
        if self._lower_upper is None:
            self._lower_upper = np.concatenate( (self.bounds ))
        return self._lower_upper
    lower_upper = property(_get_lower_upper)


    def _get_center(self):
        if self._center is None:
            bounds = self._get_bounds()
            self._center = np.mean(bounds, axis=0) 
        return self._center 
    center = property(_get_center)

    def _get_extent(self):
        if self._extent is None:
            dimensions = self._get_dimensions()
            self._extent = np.max(dimensions)/2.
        return self._extent 
    extent = property(_get_extent)
        
    def _get_dimensions(self):
        if self._dimensions is None:
            bounds = self._get_bounds()
            self._dimensions = bounds[1]-bounds[0]
        return self._dimensions
    dimensions = property(_get_dimensions)

    def _get_bounds_extent(self):
        if self._bounds_extent is None:
            lower, upper = np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
            dimensions = upper - lower
            extent = np.max(dimensions)/2.
            self._bounds_extent = lower, upper, extent
        return self._bounds_extent
    bounds_extent = property(_get_bounds_extent)   

    def _get_center_extent(self):
        if self._center_extent is None:
            vertices = self.vertices
            bounds = np.min(vertices, axis=0), np.max(vertices, axis=0)
            center = np.mean(bounds, axis=0)
            dimensions = bounds[1] - bounds[0]
            extent = np.max(dimensions)/2.
            self._center_extent = center, extent
        return self._center_extent
    center_extent = property(_get_center_extent)   

    def _get_model2world(self):
        center, extent = self.center_extent
        return ModelToWorld(extent, center)
    model2world = property(_get_model2world)

    def _get_world2model(self):
        center, extent = self.center_extent
        return WorldToModel(extent, center)
    world2model = property(_get_world2model)
 
    def __str__(self):
        lower, upper = self.bounds
        dimensions = upper - lower
        return "\n".join([
                  "upper      %s " % upper,
                  "center     %s " % np.mean([lower,upper], axis=0),
                  "lower      %s " % lower,
                  "dimensions %s " % dimensions,
                  "extent     %s " % str(np.max(dimensions)/2.),
                   ])

    def __repr__(self):
        return " ".join( [
                           "%s" % self.__class__.__name__,
                           "v %s " % len(self._vertices), 
                           "t %s " % len(self._triangles), 
                           "n %s " % len(self._normals), 
                           ])




class DAECompositeMesh(DAEMesh):
    """
    A DAEMesh that remembers which of its vertices, triangles and normals
    belongs to constituent meshes and can extract sub-meshes for them.
    Created in DAEGeometry.flatten 
    """
    def __init__(self, vertices, triangles=[], normals=[], nv=[], nt=[], nn=[], bbox=[], ids=[], indices=[], channels=[]):
        DAEMesh.__init__(self, vertices, triangles, normals )
        assert len(nv) == len(nt) == len(nn) == len(bbox) + 1 == len(indices) + 1 == len(channels) + 1, "length mismatch"
        self.size = len(nv) - 1 
        self.nv = nv
        self.nt = nt
        self.nn = nn
        self.bbox = bbox
        self.ids = ids
        self.indices = indices
        self.channels = channels

    def save(self, path_ ):
        """
        :param path_: function that takes a name argument and returns an absolute path
        """
        np.save(path_("vertices"), self._vertices )
        np.save(path_("triangles"), self._triangles )
        np.save(path_("normals"), self._normals )
        np.save(path_("nv"), self.nv )
        np.save(path_("nt"), self.nt )
        np.save(path_("nn"), self.nn )
        np.save(path_("bbox"), self.bbox )
        np.save(path_("ids"), self.ids )
        np.save(path_("indices"), self.indices )
        np.save(path_("channels"), self.channels )

    @classmethod
    def load(cls, path_): 
        vertices = np.load(path_("vertices"))
        triangles = np.load(path_("triangles"))
        normals = np.load(path_("normals"))
        nv = np.load(path_("nv"))
        nt = np.load(path_("nt"))
        nn = np.load(path_("nn"))
        bbox = np.load(path_("bbox"))
        ids = np.load(path_("ids"))
        indices = np.load(path_("indices"))
        channels = np.load(path_("channels"))
        mesh = DAECompositeMesh(vertices, triangles, normals, nv, nt, nn, bbox, ids, indices, channels)
        log.debug(mesh)
        return mesh

    def _set_index(self, index):
        if index >= self.size or index < -1:
            raise IndexError 

        if not self._index == index:
            self.reset()
            self._index = index
        pass
    def _get_index(self):
        return self._index
    index = property(_get_index,_set_index)



    def nodeindex_to_index(self, nodeindex):
        """
        :param nodeindex: geometry nodeindex, eg 3153
        :return index: internal index into the DAECompositeMesh data structures

        Translate nodeindex into internal index, giving -1 if the lookup fails.
        """
        try:
            index = np.where(self.indices == nodeindex)[0][0]
        except IndexError:
            index = -1
            log.warn("nodeindex %s is not present within the %s contained indices %s " % (nodeindex, len(self.indices), repr(self.indices)))
        pass
        return index 

    def _set_nodeindex(self, nodeindex):
        index = self.nodeindex_to_index(nodeindex) 
        self._index = index 
    def _get_nodeindex(self):
        if self._index < 0:return -1
        return self.indices[self._index]
    nodeindex = property(_get_nodeindex,_set_nodeindex)





    def _get_vertices(self):
        i = self._index
        if i == -1:return self._vertices
        nv = self.nv
        return self._vertices[nv[i]:nv[i+1]]

    def _get_triangles(self):
        i = self._index
        if i == -1:return self._triangles
        nt = self.nt
        nv = self.nv
        return self._triangles[nt[i]:nt[i+1]] - nv[i]   ## un-offsetting vertex indices 

    def _get_normals(self):
        i = self._index
        if i == -1:return self._normals
        nn = self.nn
        return self._normals[nn[i]:nn[i+1]]

    def _get_sid(self):
        i = self._index
        if i == -1:return None
        return self.ids[i]

    def _get_schannel(self):
        i = self._index
        if i == -1:return None
        return self.channels[i]

    def _get_sindice(self):
        i = self._index
        if i == -1:return None
        return self.indices[i]

    def _get_sbbox(self):
        i = self._index
        if i == -1:return None
        return self.bbox[i]

    def _get_subsolid(self):
        return DAESubSolid(self.vertices.copy(), self.triangles.copy(), self.normals.copy(), self.sindice, self.sid, self.schannel)


    # getters that return sub-ranges of underlying arrays based on value of index property
    vertices = property(_get_vertices)
    triangles = property(_get_triangles)
    normals = property(_get_normals)
    sid     = property(_get_sid)
    schannel = property(_get_schannel)
    sindice = property(_get_sindice)
    sbbox   = property(_get_sbbox)
    subsolid = property(_get_subsolid)



    def __len__(self):
        return self.size 

    def __getitem__(self, index):
        """
        :param nodeindex:
        :return subsolid item: 

        Item contains a copy of vertices, triangles and normals 
        corresponding to subsolid indicated by the index.
        """
        if index < 0:
            index = self.size + index

        if index > self.size:
            raise IndexError

        prior = self.index
        self.index = index
        subsolid = self.subsolid
        self.index = prior
        return subsolid


    def indices_for_regexp(self, ptn):
        """
        :param ptn: regexp string
        :return: list of internal indices of matching subsolids
        """
        ptn = re.compile(ptn)
        ids = self.ids
        vsearch = np.vectorize(lambda x:bool(ptn.search(x)))
        msk = vsearch(ids)               # ndarray of bool
        idx = np.where(msk == True)[0]   # array of indices
        return idx 





class DAESubSolid(DAEMesh):
    def __init__(self, vertices, triangles, normals, sindice, sid, schannel):
        DAEMesh.__init__(self, vertices, triangles, normals)
        self.index = sindice   # originating node index
        self.id = sid          # name string, truncated at some length 
        self.channel = schannel

    def __repr__(self):
        # py26 needs the positional indices 0,1,2    py27 doesnt 
        return "{0:6.1f} 0x{1:-7x} {2:-5d}  {3:s}".format(self.extent, self.channel, self.index, self.id) 
    __str__ = __repr__
 

class DAESolid(DAEMesh):
    """
    Without re-generating normals are getting more normals than 
    vertices but less than triangles ? Maybe due to triangle 
    from quad generation done by pycollada.

    DAEMesh vertex 466  triangles 884  normals 584 
    """
    secs = {}
    def __init__(self, node, bound=True, generateNormals=True):
        """
        :param node: DAENode instance 
        """
        assert node.__class__.__name__ == 'DAENode'
        pl = list(node.boundgeom.primitives())[0] if bound else node.geo.geometry.primitives[0]

        #tris = pl.triangleset()
        tris = self.make_tris(pl) 

        if generateNormals:
            self.genNormals(tris)
       
        DAEMesh.__init__(self, tris._vertex, tris._vertex_index, tris._normal )

        self.index = node.index
        self.id = node.id
        self.channel = node.channel
        self.node = node   

        if not self.check():
            log.debug("DAESolid Meshcheck failure %s " % self)

    @timing(secs)
    def make_tris(self, pl):
        return pl.triangleset()

    @timing(secs)
    def genNormals(self, tris):
        tris.generateNormals()


    def __repr__(self):
        """
        String repr used on clicking point on OpenGL window to list containing solids
        """
        return "{0:6.1f} {1:-5d}  {2:s}".format(self.extent, self.index, self.id)    

    __str__ = __repr__
       

class DAEGeometry(object):
    """
    DAEGeometry 
    __init__        :      6.395          1      6.395     ## getall(=parse) + make_solids = 2.462 + 3.933 = 6.395 
    flatten         :      0.132          1      0.132 
    make_solids     :      3.933          1      3.933 

    DAENode    
    getall          :      2.462          1      2.462 
    init            :      2.460          1      2.460 
    parse           :      2.460          1      2.460 

    DAESolid   
    genNormals      :      2.763       9068      0.000 
    make_tris       :      0.401       9068      0.000 

        2.763+0.401 = 3.164


    DAEMesh    
    __init__        :      0.022       9069      0.000 



    """
    secs = {}
    @timing(secs)
    def __init__(self, config, fromcache=False ):
        """
        :param config: DAEConfig instance
        """
        self.config = config
        if not fromcache:
            self.solids = self.get_solids(config)
            self.mesh = None
        else:
            geocachepath = config.geocachepath
            assert os.path.exists(geocachepath), geocachepath 
            log.info("populate_from_cache %s " % geocachepath )
            self.populate_from_cache(geocachepath)
        pass


    def populate_from_cache(self, cachedir):
        cachedir = os.path.join(cachedir, "daegeometry")
        npy_ = lambda name:os.path.join(cachedir, "%s.npy" % name)
        self.mesh = DAECompositeMesh.load(npy_)

    def save_to_cache(self, cachedir):
        cachedir = os.path.join(cachedir, "daegeometry")
        if not os.path.exists(cachedir):
            os.makedirs(cachedir) 
        npy_ = lambda name:os.path.join(cachedir, "%s.npy" % name)
        self.mesh.save(npy_)


    @timing(secs)
    def get_solids(self, config):
        nodespec = self.resolve_geometry_arg( config )
        if nodespec is None:
            solids = []
        else:
            o_nodes = DAENode.getall(nodespec, config.path)
            if not config.args.regexp is None:
                regexp = Regexp(config.args.regexp)
                nodes = regexp.select(o_nodes)
                log.info("geometry regexp %s reduced nodes from %s to %s " % (config.args.regexp,len(o_nodes), len(nodes)))
            else:
                nodes = o_nodes
            pass
            solids = self.make_solids(nodes, config.args.bound)
        pass
        return solids
 
    @timing(secs)
    def make_solids(self, nodes, bound):
        """
        :param nodes: list of DAENode instances
        :param bound: almost always True, 
                      when False pycollada local coordinates are used (for low level internal testing)


        :return: python list of many thousands of DAESolids instances

        Taking almost 4 seconds 

        TODO: work out how the solid info is being used and replace with 
              a single ndarray structure for all solids

        """
        return [DAESolid(node, bound) for node in nodes]

 
    def __str__(self):
        return "-p %s -g %s " % ( self.config.args.path, self.config.args.geometry )

    def resolve_geometry_arg(self, config ):
        """
        Raw Geometry arguments like "3154:" are returned unchanged.

        The default geometry config argument of "DAE_GEOMETRY_%(path)s" signals
        that detector specific geometry node specification is to be resolved
        from envvars such as DAE_GEOMETRY_DYB and DAE_GEOMETRY_JUNO
        Using the uppercased path config argument.
        """
        arg = config.args.geometry
        if not arg.startswith('DAE_'):
            return arg
        pass
        envvar = (arg % dict(path=config.args.path)).upper() 
        nodespec = os.environ.get(envvar,None) 
        log.info("resolve_geometry_arg %s to envvar %s yielding nodespec %s " % (arg, envvar, nodespec )) 
        assert nodespec
        return nodespec 

    def nodes(self):
        """
        :return: list of DAENode instances

        This does not survive the cache
        """
        #traceback.print_stack()
        return [solid.node for solid in self.solids] 

    def find_solid(self, target ):
        """
        :param target:

        Find by solid by relative indexing into the list of solids loaded 
        where the target argument begins with "-" or "+". 
        Otherwise find by the absolute geometry node index of the target.
        """
        target = str(target)
        index = None
        solid = None

        if target == "..":                  # entire mesh 
            self.mesh.index = -1
            solid = self.mesh  
            #solid = self.oldmesh  
        elif target[0] == "+" or target[0] == "-":            # relative addressing 
            solid = self.find_solid_by_index(int(target)) 
        else:                                               # absolute addressing
            index = self.mesh.nodeindex_to_index(int(target))
            if index != -1:
                solid = self.find_solid_by_index(index) 
            pass
        pass
        log.info("find_solid target %s => index %s => solid %s  " % (target, index, repr(solid)))
        return solid

 
    def find_solid_by_index(self, index):
        """
        Used by DAEViewpoint to support bookmarks, the "DAESolid" 
        object returned needs to provide: index, extent, model2world, world2model

        :para index:
        """
        try:
            return self.mesh[index]
        except IndexError:
            return None

    def find_solids_by_indices(self, indices):
        """
        :param indices: internal indices into the "solids list" 
                        actually DAECompositeMesh 

        :return: list of DAESubSolid instances, or None for invalid indices
        """
        solids = []
        for index in indices:
            solids.append(self.find_solid_by_index(index))
        return solids

    def find_solids_by_regexp(self, ptn):
        """
        g.find_solids_by_regexp("PmtHemiCathode")
        """
        indices = self.mesh.indices_for_regexp(ptn)
        return self.find_solids_by_indices(indices)

    @classmethod
    def load_from_cache(cls, config):
        obj = cls(config, fromcache=True)
        return obj

    @classmethod
    def get(cls, config):
        """
        Get from cache when `geocache` configured or create  
        """
        log.debug("DAEGeometry.get START")
        geocachepath = config.geocachepath
        if config.args.geocache and not os.path.exists(geocachepath):
            log.warn("geocache was requested by no cache exists at %s : will create the cache" % geocachepath )
        pass
        if os.path.exists(geocachepath) and config.args.geocache:
            geometry = cls.load_from_cache( config )
        else:
            geometry = cls(config)
            geometry.flatten()
            if config.args.geocache or config.args.geocacheupdate:
                geometry.save_to_cache(geocachepath)
            pass

        #log.info("DAEGeometry.get DONE timing_report ")
        #timing_report([DAEGeometry, DAENode, DAESolid, DAEMesh])
        return geometry 

    @timing(secs)
    def flatten(self):
        """  
        Adapted from Chroma geometry flattening 

        Converts from pycollada internal numpy storage into contiguous 
        arrays ready to be placed into an OpenGL VBO (Vertex Buffer Object).
        """

        nv = np.cumsum([0] + [len(solid.vertices) for solid in self.solids])
        nt = np.cumsum([0] + [len(solid.triangles) for solid in self.solids])
        nn = np.cumsum([0] + [len(solid.normals) for solid in self.solids])

        vertices = np.empty((nv[-1],3), dtype=np.float32)
        triangles = np.empty((nt[-1],3), dtype=np.uint32)
        normals = np.empty((nn[-1],3), dtype=np.float32)

        for i, solid in enumerate(self.solids):
            vertices[nv[i]:nv[i+1]] = solid.vertices
            triangles[nt[i]:nt[i+1]] = solid.triangles + nv[i]   # NB offseting vertex indices
            normals[nn[i]:nn[i+1]] = solid.normals

        idmaxlen = 100
        ids = np.empty((len(self.solids),),np.dtype((np.str_,idmaxlen)))
        bbox = np.empty((len(self.solids),6))    
        indices = np.empty((len(self.solids),),dtype=np.uint32)    
        channels = np.empty((len(self.solids),),dtype=np.uint32)    

        for i, solid in enumerate(self.solids):
            bbox[i] = solid.lower_upper
            ids[i] = solid.id   # tail chars > idmaxlen are truncated
            indices[i] = solid.index
            channels[i] = solid.channel


        log.debug('Flattening %s DAESolid into one DAEMesh...' % len(self.solids))
        assert len(self.solids) > 0, "failed to find solids, MAYBE EXCLUDED BY -g/--geometry option ? try \"-g 0:\" or \"-g 1:\" "

        mesh = DAECompositeMesh(vertices, triangles, normals, nv, nt, nn, bbox, ids, indices, channels)
        
        log.info("flatten nsolids %s into mesh: %s " % (len(self.solids),repr(mesh)))
        self.mesh = mesh 

    #def _get_oldmesh(self):
    #    return DAEMesh(self.mesh._vertices, self.mesh._triangles, self.mesh._normals)        
    #oldmesh = property(_get_oldmesh)

    def containing_solids(self, xyz ):
        """
        Find solids that contain the world frame coordinates argument,  
        sorted by extent.
        """
        indices = self.find_bbox_solid( xyz )
        solids = self.find_solids_by_indices(indices)
        solids = sorted(solids, key=lambda _:_.extent)
        pass
        return solids 


    def find_bbox_solid(self, xyz):
        """
        :param xyz: world frame coordinate

        Used by DAEScene.clicked_point

        Find indices of all solids that contain the world frame coordinate provided  
        """
        x,y,z = xyz 
        b = self.mesh.bbox
        f = np.where(
              np.logical_and(
                np.logical_and( 
                  np.logical_and(x > b[:,0], x < b[:,3]),
                  np.logical_and(y > b[:,1], y < b[:,4]) 
                              ),  
                  np.logical_and(z > b[:,2], z < b[:,5])
                            )   
                    )[0]
        return f

    def find_bbox_solid_slowly(self, xyz):
        x,y,z = xyz
        f = [] 
        def unprefix(s): 
            prefix = "__dd__Geometry__"
            return s[len(prefix):] if s[0:len(prefix)] == prefix else s 
        for i, solid in enumerate(self.solids):
            lower, upper = solid.bounds
            with printoptions(precision=3, suppress=True, strip_zeros=False):
                inside = lower[0] < x < upper[0] and lower[1] < y < upper[1] and lower[2] < z < upper[2]
                marker = "*" if inside else "-"
                if inside:
                    f.append(i)
                    print "%-5d %-5d [%s] %-50s %-40s %-40s (%7.3f) %-40s " % ( i, solid.index, marker, unprefix(solid.id), lower, upper, solid.extent, solid.dimensions )
        return f 

    def make_vbo(self,scale=False, rgba=(0.7,0.7,0.7,0.5)):
        if self.mesh is None:
            self.flatten() 

        self.mesh.index = -1
        if scale:
            vertices = (self.mesh.vertices - self.mesh.center)/self.mesh.extent
        else:
            vertices = self.mesh.vertices
        return DAEVertexBufferObject(vertices, self.mesh.normals, self.mesh.triangles, rgba )
      
    @timing(secs)
    def make_chroma_geometry(self, bvh=True):
        """
        This was formerly converting the entire geometry, not the 
        selection of solids that are used to create the VBO.
        As the former root_index based attempt to skip nodes
        was being ignored, resulting in huge universe.

        Potentially the huge universe may have bad impact on chroma BVH morton codes, 
        as most of the morton space was empty.
        """
        log.info("make_chroma_geometry bvh %s " % (bvh) )
        from env.geant4.geometry.collada.collada_to_chroma  import ColladaToChroma 

        cc = ColladaToChroma(DAENode, bvh=bvh )     
        cc.convert_geometry(nodes=self.nodes())

        self.cc = cc
        self.chroma_material_map = DAEChromaMaterialMap( self.config, cc.cmm )
        self.chroma_material_map.write()
        log.debug("completed DAEChromaMaterialMap.write")

        self.chroma_surface_map = DAEChromaSurfaceMap( self.config, cc.csm )
        self.chroma_surface_map.write()
        log.debug("completed DAEChromaSurfaceMap.write")

        cpm = self.make_chroma_process_map()
        self.chroma_process_map = DAEChromaProcessMap( self.config, cpm )
        self.chroma_process_map.write()
        log.debug("completed DAEChromaProcessMap.write")

        log.info("completed make_chroma_geometry")
        return cc.chroma_geometry


    def make_chroma_process_map(self):
        """
        Return dict of process names keyed by enum integer codes 
        """
        from photons import PHOTON_FLAGS
        cpm = {}
        for name, code in PHOTON_FLAGS.items():
            cpm[code] = name
        pass
        return cpm


    def plot( self, prop="RINDEX", materials="MineralOil LiquidScintillator GdDopedLS Acrylic".split() ):
        import matplotlib.pyplot as plt

        collada = self.cc.nodecls.orig
        cmm = dict([(shortname_(cmat.id), cmat) for cmat in collada.materials])

        for name in materials:
            if not name in cmm:
                log.warn("material named %s not one of %s " % (name, repr(cmm.keys()))) 
                continue
            pass
            mat = cmm[name] 
            plt.plot( *mat.extra.properties[prop].T, label=name) 
        pass
        plt.legend(title=prop)
        plt.show()



 

class DAEVertexBufferObject(object):
    """
    Used in g4daeview.py main::

        vbo = geometry.make_vbo(scale=scene.scaled_mode, rgba=config.rgba)
        mesh = gp.graphics.VertexBuffer( vbo.data, vbo.faces )

    The glumpy VertexBuffer looks for below att names in
    the dtype of the vertices passed in first argument.

        ==================  ==================   ========================
         gl***Pointer          GL_***_ARRAY        VertexAttribute_***
        ==================  ==================   ========================
         Color                COLOR                color  
         EdgeFlag             EDGE_FLAG            edge_flag
         FogCoord             FOG_COORD            fog_coord
         Normal               NORMAL               normal
         SecondaryColor       SECONDARY_COLOR      secondary_color
         TexCoord             TEXTURE_COORD        tex_coord
         Vertex               VERTEX               position
         VertexAttrib         N/A             
        ==================  ==================   ========================


    These are then converted into VertexAttribute_*** instances with 
    appropriate counts and types converted from the numpy dtype.

    """
    def __init__(self, vertices, normals, faces, rgba ):
        nvert = len(vertices)
        data = np.zeros(nvert, [('position', np.float32, 3), 
                                ('color',    np.float32, 4), 
                                ('normal',   np.float32, 3)])
        data['position'] = vertices
        data['color'] = np.tile( rgba, (nvert, 1))
        data['normal'] = normals

        self.data = data
        self.faces = faces

    def __repr__(self):
        with printoptions(precision=3, suppress=True, strip_zeros=False):
            return "\n".join([
                   "position",str(self.data['position']),
                   "color",str(self.data['color']),
                   "normal",str(self.data['normal']),
                   "faces",str(self.faces),
                   ])






def check_find( dg ):
    xyz = (-16632.046096412007, -796063.5921605631, -2716.5372465302394 ) 

    f = dg.find_bbox_solid(xyz)
    print "find_bbox_solid for world point %s yields solids %s " % ( str(xyz), f )

    solids = [dg.solids[_] for _ in f]
    print "\n".join([ "\n".join([repr(solid),solid.smry()]) for solid in solids])



def check_geometry( g, cg ):
    """
    Passing solid_id_map array pointer in kernel call allows 
    to map from triangle_id into solid_id. This would allow 
    picking a photon and seeing all solids that it touches.

    chroma/gpu/geometry.py:: 

       self.solid_id_map = ga.to_gpu(geometry.solid_id.astype(np.uint32))

    chroma/cuda/daq.cu::

       52     int triangle_id = last_hit_triangles[photon_id];
       53 
       54     if (triangle_id > -1) {
       55         int solid_id = solid_map[triangle_id];

    """
    assert len(cg.surface_index) == len(cg.solid_id) == len(cg.material1_index) == len(cg.material2_index) == len(cg.colors) == len(cg.mesh.triangles), "expecting solid/material indices for every triangle"
    assert cg.surface_index.max() < 40 , "expecting less than 40 surfaces "
    assert cg.material1_index.max() < 30 , "expecting less than 30 materials "
    assert cg.material2_index.max() < 30 , "expecting less than 30 materials "
    assert cg.solid_id.min() == 0 
    assert cg.solid_id.max() + 1 == len(g.solids) == len(cg.solids) , "mismatch between DAEGeometry solid count and chroma geometry "


def check_material( g, cg ):
    """
    ::

        delta:~ blyth$ daegeometry.sh
        2014-07-02 19:27:41,175 env.geant4.geometry.collada.g4daeview.daegeometry:505 creating DAEGeometry instance from DAEConfig in standard manner
        label                    absorption_length        scattering_length        refractive_index         reemission_prob          reemission_cdf          
         0 Air                   False                    False                    False                    True                     True                    
         1 Aluminium             False                    False                    False                    True                     True                    
         2 GdDopedLS             False                    False                    False                    False                    True                    
         3 Acrylic               False                    False                    False                    True                     True                    
         4 Teflon                False                    False                    False                    True                     True                    
         5 LiquidScintillator    False                    False                    False                    False                    True                    
         6 Bialkali              False                    False                    False                    True                     True                    
         7 OpaqueVacuum          False                    False                    False                    True                     True                    
         8 Vacuum                False                    False                    False                    True                     True                    
         9 Pyrex                 False                    False                    False                    True                     True                    
        10 UnstStainlessSteel    False                    False                    False                    True                     True                    
        11 PVC                   False                    False                    False                    True                     True                    
        12 StainlessSteel        False                    False                    False                    True                     True                    
        13 ESR                   False                    False                    False                    True                     True                    
        14 Nylon                 False                    False                    False                    True                     True                    
        15 MineralOil            False                    False                    False                    True                     True                    
        16 BPE                   False                    False                    False                    True                     True                    
        17 Ge_68                 False                    False                    False                    True                     True                    
        18 Co_60                 False                    False                    False                    True                     True                    
        19 C_13                  False                    False                    False                    True                     True                    
        20 Silver                False                    False                    False                    True                     True                    
        21 Nitrogen              False                    False                    False                    True                     True                    
        22 Water                 False                    False                    False                    True                     True                    
        23 NitrogenGas           False                    False                    False                    True                     True                    
        24 IwsWater              False                    False                    False                    True                     True                    
        25 ADTableStainlessSteel False                    False                    False                    True                     True                    
        26 Tyvek                 False                    False                    False                    True                     True                    
        27 OwsWater              False                    False                    False                    True                     True                    
        28 DeadWater             False                    False                    False                    True                     True                    
        2014-07-02 19:27:51,690 env.geant4.geometry.collada.g4daeview.daegeometry:524 dropping into IPython.embed() try: g.<TAB> cg.<TAB>

    """
    props = "absorption_length scattering_length refractive_index reemission_prob reemission_cdf".split()
    labels = ["%2d %s" % (i,shortname_(mat.name)) for i, mat in enumerate(cg.unique_materials)]
    maxl = str(max(map(len, props+labels)))
    fmt_ = lambda _:("%-"+maxl+"s") % str(_)
    print " ".join(map(fmt_, ["label"] + props)) 
    for label, mat in zip(labels,cg.unique_materials):
        missing_ = lambda _:np.all(getattr(mat,_)[:,1] == 0.)
        print " ".join(map(fmt_, [label] + map(missing_, props)))



def check_collada2chroma_material( cmat, props ):
    """
    Look at material properties and their use for reemission in
    NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc

    Tips for reading horrendously poorly formatted code in vim:

    #. `set fdm=syntax` "foldmethod" 
    #. open/close folds with `za` on lines with curlies

    ::

        441     G4int materialIndex = aMaterial->GetIndex();
        443     G4PhysicsOrderedFreeVector* ReemissionIntegral = NULL;
        444     ReemissionIntegral =
        445         (G4PhysicsOrderedFreeVector*)((*theReemissionIntegralTable)(materialIndex));

    DsG4Scintillation::BuildThePhysicsTable() integrates for each material `i`
    the property tables FASTCOMPONENT, SLOWCOMPONENT, REEMISSIONPROB 
    and sets the(Fast/Slow/Reemission)IntegralTable::

        967         theFastIntegralTable->insertAt(i,aPhysicsOrderedFreeVector);
        968         theSlowIntegralTable->insertAt(i,bPhysicsOrderedFreeVector);
        969         theReemissionIntegralTable->insertAt(i,cPhysicsOrderedFreeVector); 

    Constants, given a dummy table range. 

    #. ReemissionFASTTIMECONSTANT 1.5   ==> fastTimeConstant
    #. ReemissionSLOWTIMECONSTANT 1.5   ==> slowTimeConstant
    #. ReemissionYIELDRATIO       1.    ==> YieldRatio

    The TIMECONSTANTs in ns feed into ScintillationTime to control time distribution 
    of reemitted photon::

        647             if (flagReemission) {
        648                 deltaTime= pPostStepPoint->GetGlobalTime() - t0
        649                            -ScintillationTime * log( G4UniformRand() );



    Propa wavelength dependant tables:

    #. FASTCOMPONENT == SLOWCOMPONENT  
    #. REEMISSIONPROB  

    Not relevant to reemission:

    #. SCINTILLATIONYIELD         11522.
    #. RESOLUTIONSCALE            1.



    SCINTILLATIONYIELD xrange -0.0012398424468 0.0012398424468 yrange 11522.0 11522.0  
    [[     0.0012  11522.    ]
     [    -0.0012  11522.    ]]

    RESOLUTIONSCALE xrange -0.0012398424468 0.0012398424468 yrange 1.0 1.0  
    [[ 0.0012  1.    ]
     [-0.0012  1.    ]]

    ReemissionYIELDRATIO xrange -0.0012398424468 0.0012398424468 yrange 1.0 1.0  
    [[ 0.0012  1.    ]
     [-0.0012  1.    ]]

    ReemissionFASTTIMECONSTANT xrange -0.0012398424468 0.0012398424468 yrange 1.5 1.5  
    [[ 0.0012  1.5   ]
     [-0.0012  1.5   ]]

    ReemissionSLOWTIMECONSTANT xrange -0.0012398424468 0.0012398424468 yrange 1.5 1.5  
    [[ 0.0012  1.5   ]
     [-0.0012  1.5   ]]

    FASTCOMPONENT xrange 79.9898352776 799.898352776 yrange 0.0 1.0276  
    [[  79.9898    0.    ]
     [ 120.0235    0.    ]
     [ 199.9746    0.    ]
     ..., 
     [ 599.0011    0.0017]
     [ 600.0012    0.0018]
     [ 799.8984    0.    ]]

    SLOWCOMPONENT xrange 79.9898352776 799.898352776 yrange 0.0 1.0276  
    [[  79.9898    0.    ]
     [ 120.0235    0.    ]
     [ 199.9746    0.    ]
     ..., 
     [ 599.0011    0.0017]
     [ 600.0012    0.0018]
     [ 799.8984    0.    ]]

    REEMISSIONPROB xrange 79.9898352776 799.898352776 yrange 0.0 0.8022  
    [[  79.9898    0.4   ]
     [ 120.0235    0.4   ]
     [ 159.9797    0.4   ]
     ..., 
     [ 575.8273    0.0587]
     [ 712.6064    0.    ]
     [ 799.8984    0.    ]]

    """ 
    assert cmat.__class__.__module__ == 'collada.material' and cmat.__class__.__name__ == 'Material'
    assert cmat.extra.__class__.__module__ == 'env.geant4.geometry.collada.g4daenode' and cmat.extra.__class__.__name__ == 'MaterialProperties'
    assert cmat.extra.properties.__class__ == dict 
    d = cmat.extra.properties
    for k in props:
        a = d.get(k,None) 
        print "%s xrange %s %s yrange %s %s  " % (k,  a[:,0].min(), a[:,0].max(), a[:,1].min(), a[:,1].max())
        print  a

    assert np.all( d['FASTCOMPONENT'] == d['SLOWCOMPONENT'] )
    assert np.all( d['ReemissionFASTTIMECONSTANT'] == d['ReemissionSLOWTIMECONSTANT'] )


def compare_materials( collada, props, a, b  ):
    cmm = dict([(shortname_(cmat.id), cmat) for cmat in collada.materials])
    print "compare_collada_material %s %s " % ( a, b )
    for k in props:
        print "%-30s %s " % ( k, np.all( cmm[a].extra.properties[k] == cmm[b].extra.properties[k] ))


def dump_extra( cmat ):
    """
    Dump the COLLADA DAE XML `extra` element. 

    #. `extra` element contains paired `matrix` and `property` elements  

    :param cmat: collada material instance

    ::

        In [3]: print tostring_(gdls.extra)
        <extra xmlns="http://www.collada.org/2005/11/COLLADASchema">

            <matrix coldim="2" name="ABSLENGTH0xc2a92f8">
                      1.3778e-06 299.6 
                      1.3793e-06 306.2 
                      1.3808e-06 328.4 
                      1.3824e-06 363.1
                      ... 
            </matrix>
            <property name="ABSLENGTH" ref="ABSLENGTH0xc2a92f8"/>

            <matrix coldim="2" name="AlphaFASTTIMECONSTANT0xbf6c870">-1 1 1 1</matrix>
            <property name="AlphaFASTTIMECONSTANT" ref="AlphaFASTTIMECONSTANT0xbf6c870"/>

            ...

            <matrix coldim="2" name="SLOWTIMECONSTANT0xbf6b638">-1 12.2 1 12.2</matrix>
            <property name="SLOWTIMECONSTANT" ref="SLOWTIMECONSTANT0xbf6b638"/>

            <matrix coldim="2" name="YIELDRATIO0xbf6b6a8">-1 0.86 1 0.86</matrix>
            <property name="YIELDRATIO" ref="YIELDRATIO0xbf6b6a8"/>
          </extra>


    Property and matrix elements are written by `env/geant4/geometry/DAE/src/G4DAEWrite.cc`:

    * `void G4DAEWrite::PropertyWrite(xercesc::DOMElement* extraElement,  const G4MaterialPropertiesTable* const ptable)`
    * `void G4DAEWrite::PropertyVectorWrite(const G4String& key,const G4MaterialPropertyVector* const pvec,xercesc::DOMElement* extraElement)`

    Invoked by `env/geant4/geometry/DAE/src/G4DAEWriteMaterials.cc`:

    * `void G4DAEWriteMaterials::MaterialWrite(const G4Material* const materialPtr)`

    """
    return tostring_(cmat.extra)



def check_props(geometry, chroma_geometry):
    check_geometry( geometry, chroma_geometry )
    check_material( geometry, chroma_geometry )

    g = geometry  
    cg = chroma_geometry 
    mm = dict([(shortname_(m.name),m) for m in cg.unique_materials])

    collada = geometry.cc.nodecls.orig
    assert collada.__class__.__name__ == 'Collada'
    cmm = dict([(shortname_(cmat.id), cmat) for cmat in collada.materials])
    
    props = "SCINTILLATIONYIELD RESOLUTIONSCALE ReemissionFASTTIMECONSTANT ReemissionSLOWTIMECONSTANT ReemissionYIELDRATIO FASTCOMPONENT SLOWCOMPONENT REEMISSIONPROB".split()
    for name in 'GdDopedLS LiquidScintillator'.split():
        print name 
        check_collada2chroma_material( cmm[name], props)
    pass
    compare_materials( collada, props, 'GdDopedLS', 'LiquidScintillator') 

    oil,ls,gdls = map(lambda _:cmm[_],'MineralOil LiquidScintillator GdDopedLS'.split())
    gdls_props = gdls.extra.properties


    chroma_geometry.save(["/tmp/tt"])


def main():
    """
    Make plot comparing properties using embedded ipython::

         mats = "MineralOil LiquidScintillator GdDopedLS Acrylic".split()
         g.plot(    "RINDEX", mats )
         g.plot( "ABSLENGTH", mats )
         g.plot(  "RAYLEIGH", mats )

    """
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()
    np.set_printoptions(precision=4, suppress=True, threshold=20)


    geometry = DAEGeometry.get(config)


    ids = geometry.mesh.ids

    ptn = re.compile(config.args.regexp)
    vsearch = np.vectorize(lambda x:bool(ptn.search(x)))
    msk = vsearch(ids)               # ndarray of bool
    idx = np.where(msk == True)[0]   # array of indices
    sel = ids[idx]                   # selection of the ids


    if config.args.ipython:
        import IPython 
        IPython.embed()
         



          

if __name__ == '__main__':
    main()




