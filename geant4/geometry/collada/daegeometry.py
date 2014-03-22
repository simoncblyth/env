#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)
import numpy as np

from env.geant4.geometry.collada.daenode import DAENode 

class DAESolid(object):
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
       
        self.tris = tris
        self.id = node.id

    def __repr__(self):
        return " ".join( [
                           "%s %s " % (self.__class__.__name__, self.id),
                           "vertex %s " % len(self.tris._vertex), 
                           "triangles %s " % len(self.tris._vertex_index), 
                           "normals %s " % len(self.tris._normal), 
                           ])

class DAEMesh(object):
    def __init__(self, vertices, triangles, normals ):
        self.check(vertices, triangles)
        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals

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


    def check(self, vertices, triangles):
        assert np.min(triangles) == 0
        assert np.max(triangles) == len(vertices)-1 , (np.max(triangles), len(vertices)-1 )

    def __repr__(self):
        return " ".join( [
                           "%s" % self.__class__.__name__,
                           "vertex %s " % len(self.vertices), 
                           "triangles %s " % len(self.triangles), 
                           "normals %s " % len(self.normals), 
                           ])

class DAEGeometry(object):
    def __init__(self, arg, path=None, bound=True):

        if path is None:
            path = os.environ['DAE_NAME']
        if len(DAENode.registry) == 0:
            DAENode.parse(path)

        self.solids = [DAESolid(node, bound) for node in DAENode.getall(arg)]
        self.mesh = None

    def flatten(self):
        """  
        Adapted from Chroma geometry flattening 
        """
        nv = np.cumsum([0] + [len(solid.tris._vertex) for solid in self.solids])
        nt = np.cumsum([0] + [len(solid.tris._vertex_index) for solid in self.solids])
        nn = np.cumsum([0] + [len(solid.tris._normal) for solid in self.solids])

        vertices = np.empty((nv[-1],3), dtype=np.float32)
        triangles = np.empty((nt[-1],3), dtype=np.uint32)
        normals = np.empty((nn[-1],3), dtype=np.float32)

        for i, solid in enumerate(self.solids):
            vertices[nv[i]:nv[i+1]] = solid.tris._vertex
            triangles[nt[i]:nt[i+1]] = solid.tris._vertex_index + nv[i]   # NB offseting vertex indices
            normals[nn[i]:nn[i+1]] = solid.tris._normal

        log.info('Flattening %s DAESolid into one DAEMesh...' % len(self.solids))
        mesh = DAEMesh(vertices, triangles, normals)
        log.info(mesh)

        self.mesh = mesh 
       


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dg = DAEGeometry("3166:3180")
    dg.flatten()





