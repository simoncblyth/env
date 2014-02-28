#!/usr/bin/env python
"""
::

   ipython demo_collada_to_chroma.py demo.dae -i 

"""
import os, sys, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # chroma has weird logging, forcing this placement 

from env.geant4.geometry.collada.daenode import DAENode
from chroma.geometry import Mesh, Solid, Material, Surface, Geometry

class ColladaToChroma(object):
    def __init__(self):
        self.geo = Geometry(detector_material=None)    # bialkali ?
        self.vcount = 0
 
    def visit(self, node, debug=False):
        self.vcount += 1
        bps = list(node.boundgeom.primitives())
        bpl = bps[0]
        assert len(bps) == 1 and bpl.__class__.__name__ == 'BoundPolylist'
        tris = bpl.triangleset()

        #print node.id
        #
        #if node.id in DAENode.extra.volmap:
        #    print "node.id %s in volmap" % node.id
        #    print DAENode.extra.volmap[node.id]


        # collada meets chroma
        vertices = tris._vertex
        triangles = tris._vertex_index
        mesh = Mesh( vertices, triangles, remove_duplicate_vertices=False ) 

        # outer/inner ? triangle orientation ?
        material1 = None  
        material2 = None

        # G4DAE persists the below surface elements which 
        # both reference "opticalsurface" containing the keyed properties
        #
        # * "skinsurface" (single volumeref) 
        # * "boundarysurface" (volumeref pair) 
        #    Are the pairs always parent/child nodes ? Attempt to find them here
        #
        #
        surface = None    

        color = 0x33ffffff 
        solid = Solid( mesh, material1, material2, surface, color )

        if debug and self.vcount % 1000 == 0:
            print node.id
            print self.vcount, bpl, tris, tris.material
            print mesh
            #print mesh.assemble()
            bounds =  mesh.get_bounds()
            extent = bounds[1] - bounds[0]
            print extent





if __name__ == '__main__':
   if len(sys.argv) > 1:
       path = sys.argv[1]
   else:     
       path = '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'
   pass    

   DAENode.parse(path)

   DAENode.extra.dump_skinsurface()
   DAENode.extra.dump_bordersurface()

   #DAENode.dump_extra_material()
  
   #cc = ColladaToChroma()
   #DAENode.vwalk(cc.visit)

   #print "volmap:"
   #print "\n".join(DAENode.extra.volmap.keys())










