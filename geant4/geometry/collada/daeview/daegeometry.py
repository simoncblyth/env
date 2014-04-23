#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)
import numpy as np
from env.geant4.geometry.collada.daenode import DAENode 

from daeutil import printoptions, ModelToWorld, WorldToModel
from daeviewpoint import DAEViewpoint


class DAEMesh(object):
    """
    TODO: remove use of duplicating pair properties, 
          now that caching in place these is no need for them
    """
    def __init__(self, vertices, triangles, normals=[] ):
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
    def __init__(self, arg, path=None, bound=True):
        """
        :param arg:  specifications of the DAE nodes to load, via index or id
        :param path: to the dae file
        :param bound: use world space coordinates
        """
        if path is None:
            path = os.environ['DAE_NAME']
        if len(DAENode.registry) == 0:
            DAENode.parse(path)

        self.solids = [DAESolid(node, bound) for node in DAENode.getall(arg)]
        self.mesh = None
        self.bbox_cache = None

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

        log.info('Flattening %s DAESolid into one DAEMesh...' % len(self.solids))
        mesh = DAEMesh(vertices, triangles, normals)
        log.info(mesh)
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
        log.info("make_chroma_geometry bvh %s " % (bvh) )
        from env.geant4.geometry.collada.collada_to_chroma  import ColladaToChroma 

        cc = ColladaToChroma(DAENode, bvh=bvh )     
        cc.convert_geometry(nodes=self.nodes())

        log.info("completed make_chroma_geometry")
        return cc.chroma_geometry


 

class DAEVertexBufferObject(object):
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






if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

    nodes = "3153:12230"
    #nodes = "3153:3200"

    dg = DAEGeometry(nodes)
    dg.flatten()
       
    xyz = (-16632.046096412007, -796063.5921605631, -2716.5372465302394 ) 

    f = dg.find_bbox_solid(xyz)
    print "find_bbox_solid for world point %s yields solids %s " % ( str(xyz), f )

    solids = [dg.solids[_] for _ in f]
    print "\n".join([ "\n".join([repr(solid),solid.smry()]) for solid in solids])


           






