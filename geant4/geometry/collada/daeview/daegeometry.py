#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)
import numpy as np
from env.geant4.geometry.collada.daenode import DAENode 

from daeutil import printoptions, ModelToWorldTransform, WorldToModelTransform
from daeviewpoint import DAEViewpoint


class DAEMesh(object):
    def __init__(self, vertices, triangles, normals=[] ):
        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals

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
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
    bounds = property(_get_bounds)

    def _get_center(self):
        bounds = self._get_bounds()
        return np.mean(bounds, axis=0) 
    center = property(_get_center)

    def _get_extent(self):
        dimensions = self._get_dimensions()
        extent = np.max(dimensions)/2.
        return extent 
    extent = property(_get_extent)
        
    def _get_dimensions(self):
        bounds = self._get_bounds()
        return bounds[1]-bounds[0]
    dimensions = property(_get_dimensions)

    def _get_bounds_extent(self):
        lower, upper = np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
        dimensions = upper - lower
        extent = np.max(dimensions)/2.
        return lower, upper, extent
    bounds_extent = property(_get_bounds_extent)   


    def _get_center_extent(self):
        bounds = np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)
        center = np.mean(bounds, axis=0)
        dimensions = bounds[1] - bounds[0]
        extent = np.max(dimensions)/2.
        return center, extent
    center_extent = property(_get_center_extent)   

    def _get_model2world(self):
        center, extent = self.center_extent
        return ModelToWorldTransform(extent, center)
    model2world = property(_get_model2world)

    def _get_world2model(self):
        center, extent = self.center_extent
        return WorldToModelTransform(extent, center)
    world2model = property(_get_world2model)
 
    def smry(self):
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
        if bound:
            pl = list(node.boundgeom.primitives())[0] 
        else:
            pl = node.geo.geometry.primitives[0]

        tris = pl.triangleset()
        if generateNormals:
            tris.generateNormals()
       
        DAEMesh.__init__(self, tris._vertex, tris._vertex_index, tris._normal )

        #self.tris = tris
        self.index = node.index
        self.id = node.id
        if not self.check():
            log.debug("DAESolid Meshcheck failure %s " % self)


    def __repr__(self):
        return " ".join([
                           DAEMesh.__repr__(self),
                           " : %s %s  " % (self.index, self.id), 
                         ])


class DAEGeometry(object):
    def __init__(self, arg, path=None, bound=True):

        if path is None:
            path = os.environ['DAE_NAME']
        if len(DAENode.registry) == 0:
            DAENode.parse(path)

        self.solids = [DAESolid(node, bound) for node in DAENode.getall(arg)]
        self.mesh = None

    def find_solid(self, target ):
        """
        Find by solid by relative indexing into the list of solids loaded 
        where the target argument begins with "-" or "+". Otherwise
        find by the absolute geometry index of the target.
        """
        if target is None or target == "":return self.mesh
        if target[0] == "+" or target[0] == "-": 
            relative = int(target)
            log.debug("relative target index %s " % relative )
            return self.solids[relative] 
        else:
            return self.find_solid_by_index(target)
            
    def find_solid_by_index(self, index):
        selection = filter(lambda _:str(_.index) == index, self.solids)
        if len(selection) == 1:
            focus = selection[0]
        else:
            focus = None
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

    def make_vbo(self,scale=False, rgba=(0.7,0.7,0.7,0.5)):
        if self.mesh is None:
            self.flatten() 
        if scale:
            vertices = (self.mesh.vertices - self.mesh.center)/self.mesh.extent
        else:
            vertices = self.mesh.vertices
        return DAEVertexBufferObject(vertices, self.mesh.normals, self.mesh.triangles, rgba )
       


    def interpret_vspec( self, vspec, args ):
        """
        Interpret view specification strings like the below into
        target, eye, look, up::

           2000
           +0
           +1
           -1
           2000_0,1,1
           2000_-1,-1,-1
           2000_-1,-1,-1
           2000_-1,-1,-1_0,0,1
           2000_-1,-1,-1_0,0,1_0,0,1

        """
        target = None
        eye = args.eye
        look = args.look
        up = args.up
   
        velem = vspec.split("_") if not vspec is None else [] 
        fvec_ = lambda _:map(float, _.split(","))

        nelem = len(velem)
        if nelem > 0: 
            target = velem[0]
        if nelem > 1:
            eye = fvec_(velem[1])
        if nelem > 2:
            look = fvec_(velem[2])
        if nelem > 3:
            up = fvec_(velem[3])

        return target, eye, look, up


    def make_view(self, vspec, args):
        """
        :param target: when None corresponds to the full mesh 

        The transform converts solid frame coordinates expressed in units of the extent
        into world frame coordinates.
        """
        target, eye, look, up = self.interpret_vspec( vspec, args )
        solid = self.find_solid(target) 
        assert solid
        return DAEViewpoint(eye, look, up, solid, target )    # use solid ?    


 

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





def check_view(geometry):
    target = "+0"
    eye = (1,1,1)
    look = (0,0,0)
    up = (0,1,0)
 
    view = geometry.make_view( target, eye, look, up)
    print view 

    view._eye = np.array([1.1,1,1])
    print view 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dg = DAEGeometry("3153:12230")
    dg.flatten()
    for i, solid in enumerate(dg.solids):
        print "%-5s %s " % ( i, solid )

   






