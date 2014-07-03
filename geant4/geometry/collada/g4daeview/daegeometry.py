#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)
import numpy as np
from env.geant4.geometry.collada.daenode import DAENode 

from daeutil import printoptions, ModelToWorld, WorldToModel
from daeviewpoint import DAEViewpoint
from collada.xmlutil import etree as ET

tostring_ = lambda _:ET.tostring(getattr(_,'xmlnode')) 
shortname_ = lambda _:_[17:-9]   # trim __dd__Materials__GdDopedLS_fx_0xc2a8ed0 into GdDopedLS 


class DAEMesh(object):
    """
    TODO: remove use of duplicating pair properties, 
          now that caching in place these is no need for them

    Note that most of the useful features of DAEMesh only 
    requires a numpy array of vertices, so a ChromaPhotonList 
    can be treated as a DAEMesh for navigation purposes.
    """
    def __init__(self, vertices, triangles=[], normals=[] ):
        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals

        self.index = -1
        self.id = "entiremesh"

        self._bounds = None
        self._lower_upper = None
        self._center = None
        self._extent = None
        self._dimensions = None
        self._bounds_extent = None
        self._center_extent = None

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
            bounds = np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
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
                           "v %s " % len(self.vertices), 
                           "t %s " % len(self.triangles), 
                           "n %s " % len(self.normals), 
                           ])





class DAESolid(DAEMesh):
    """
    Without re-generating normals are getting more normals than 
    vertices but less than triangles ? Maybe due to triangle 
    from quad generation done by pycollada.

    DAEMesh vertex 466  triangles 884  normals 584 
    """
    def __init__(self, node, bound=True, generateNormals=True):
        """
        :param node: DAENode instance 
        """
        assert node.__class__.__name__ == 'DAENode'
        pl = list(node.boundgeom.primitives())[0] if bound else node.geo.geometry.primitives[0]
        tris = pl.triangleset()
        if generateNormals:
            tris.generateNormals()
       
        DAEMesh.__init__(self, tris._vertex, tris._vertex_index, tris._normal )

        self.index = node.index
        self.id = node.id
        self.node = node   

        if not self.check():
            log.debug("DAESolid Meshcheck failure %s " % self)

    def __repr__(self):
        return "{0:6.1f} {1:-5d}  {2:s}".format(self.extent, self.index, self.id)    # py26 needs the positional indices 0,1,2    py27 doesnt 

    __str__ = __repr__
       

class DAEGeometry(object):
    def __init__(self, arg, config ):
        """
        :param arg:  specifications of the DAE nodes to load, via index or id
        :param config: 

        """
        if arg is None:
            solids = []
        else:
            path = config.path
            bound = config.args.bound
            nodes = DAENode.getall(arg, path)
            solids = [DAESolid(node, bound) for node in nodes]
        pass
        self.solids = solids
        self.mesh = None
        self.bbox_cache = None
        self.config = config
       
    def __str__(self):
        return "-p %s -g %s " % ( self.config.args.path, self.config.args.geometry )

    def nodes(self):
        """
        :return: list of DAENode instances
        """
        return [solid.node for solid in self.solids] 

    def find_solid(self, target ):
        """
        :param target:

        Find by solid by relative indexing into the list of solids loaded 
        where the target argument begins with "-" or "+". Otherwise
        find by the absolute geometry index of the target.
        """
        if target == "..":return self.mesh  # entire mesh
        if target[0] == "+" or target[0] == "-": 
            relative = int(target)
            log.debug("relative target index %s " % relative )
            return self.solids[relative] 
        else:
            return self.find_solid_by_index(target)
            
    def find_solid_by_index(self, index):
        """
        :para index:
        """
        selection = filter(lambda _:str(_.index) == index, self.solids)
        focus = selection[0] if len(selection) == 1 else None
        return focus


    def make_mesh(self, vertices, triangles, normals ):
        mesh = DAEMesh(vertices, triangles, normals)
        log.debug(mesh)
        return mesh

    def save_to_cache(self, cachepath):
        if self.bbox_cache is None:
            self.make_bbox_cache()
        pass
        if os.path.exists(cachepath):
            log.info("overwriting preexisting cache file %s " % cachepath ) 
        else:
            log.info("writing cache file %s " % cachepath ) 
        pass
        np.savez(cachepath, bbox_cache=self.bbox_cache, vertices=self.mesh.vertices, triangles=self.mesh.triangles, normals=self.mesh.normals)

    def populate_from_cache(self, npz):
        mesh = self.make_mesh( npz['vertices'], npz['triangles'], npz['normals'] )                
        self.mesh = mesh 
        self.bbox_cache = npz['bbox_cache']  

    @classmethod
    def load_from_cache(cls, config):
        geocachepath = config.geocachepath
        assert os.path.exists(geocachepath), geocachepath 
        log.info("load_from_cache %s " % geocachepath )
        npz = np.load(geocachepath)
        obj = cls(None, config)
        obj.populate_from_cache(npz)
        return obj


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

        log.debug('Flattening %s DAESolid into one DAEMesh...' % len(self.solids))

        assert len(self.solids) > 0, "failed to find solids, MAYBE EXCLUDED BY -g/--geometry option ? try \"-g 0:\" or \"-g 1:\" "

        mesh = self.make_mesh(vertices, triangles, normals)
        self.mesh = mesh 

    def make_bbox_cache(self):
        """
        """
        bbox_cache = np.empty((len(self.solids),6))    
        for i, solid in enumerate(self.solids):
            bbox_cache[i] = solid.lower_upper
        pass
        self.bbox_cache = bbox_cache

    def find_bbox_solid(self, xyz):
        """
        :param xyz: world frame coordinate

        Find indices of all solids that contain the world frame coordinate provided  
        """
        if self.bbox_cache is None:
            self.make_bbox_cache() 
        x,y,z = xyz 
        b = self.bbox_cache
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
        if scale:
            vertices = (self.mesh.vertices - self.mesh.center)/self.mesh.extent
        else:
            vertices = self.mesh.vertices
        return DAEVertexBufferObject(vertices, self.mesh.normals, self.mesh.triangles, rgba )
      
    def make_chroma_geometry(self, bvh=True):
        """
        This was formerly converting the entire geometry, not the 
        selection of solids that are used to create the VBO.
        As the former root_index based attempt to skip nodes
        was being ignored, resulting in huge universe.

        Potentially the huge universe may have bad impact on chroma BVH morton codes, 
        as most of the morton space was empty.
        """
        log.debug("make_chroma_geometry bvh %s " % (bvh) )
        from env.geant4.geometry.collada.collada_to_chroma  import ColladaToChroma 

        cc = ColladaToChroma(DAENode, bvh=bvh )     
        cc.convert_geometry(nodes=self.nodes())
        self.cc = cc

        log.debug("completed make_chroma_geometry")
        return cc.chroma_geometry


 

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
    assert cmat.extra.__class__.__module__ == 'env.geant4.geometry.collada.daenode' and cmat.extra.__class__.__name__ == 'MaterialProperties'
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



def make_plot():
    import matplotlib.pyplot as plt

    #plt.plot(sh[1][0:-1], sh[0], 'r-', rh[1][0:-1], rh[0], 'b-')



def main():
    """
    """
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()
    np.set_printoptions(precision=4, suppress=True, threshold=20)

    log.info("creating DAEGeometry instance from DAEConfig in standard manner"  )
    geometry = DAEGeometry(config.args.geometry, config)
    geometry.flatten()
    chroma_geometry = geometry.make_chroma_geometry()

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


    gdls = cmm['GdDopedLS']
    d = gdls.extra.properties


    log.info("dropping into IPython.embed() try: g.<TAB> cg.<TAB>")
    import IPython 
    IPython.embed()


          

if __name__ == '__main__':
    main()




