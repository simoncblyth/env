#!/usr/bin/env python
"""

::

   ipython demo_collada_to_chroma.py demo.dae -i 

::

    G4MaterialPropertiesTable* G4Material::GetMaterialPropertiesTable()


* :google:`GetMaterialPropertiesTable wavelength`


"""
import os, sys, logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)   # chroma has weird logging, forcing this placement 

from env.geant4.geometry.collada.daenode import DAENode
from chroma.geometry import Mesh, Solid, Material, Surface, Geometry


class CheckMaterial(object):
    def __init__(self):
        self.vcount = 0
        self.nodes_with_material = dict(TOTAL=0)
        self.refs_with_material = dict()
  
    def visit(self, node, debug=False):
        self.vcount += 1

        # counting nodes for each material symbol
        if not node.symbol in self.nodes_with_material: 
            self.nodes_with_material[node.symbol] = 0
        self.nodes_with_material[node.symbol] += 1
        self.nodes_with_material['TOTAL'] += 1

        # lists of nodes for each material symbol
        if not node.symbol in self.refs_with_material:
            self.refs_with_material[node.symbol] = []
        self.refs_with_material[node.symbol].append(node)

        if debug and self.vcount % 1000 == 0:
            print "vcnt    ", vcount
            print "node    ", node
            print "bgeo    ", node.boundgeom
            print "symbol  ", node.symbol
            print "matnode ", node.matnode
            print "matid   ", node.matid
     
    def summary(self):
        print "vcount %s " % self.vcount
        print "\n".join(["%-25s : %s " % (symbol,self.nodes_with_material[symbol]) for symbol in sorted(self.nodes_with_material,key=lambda _:self.nodes_with_material[_])]) 
        print "\n".join(["%-25s : %s " % (symbol,len(self.refs_with_material[symbol])) for symbol in sorted(self.refs_with_material,key=lambda _:len(self.refs_with_material[_]))]) 

        for symbol in sorted(self.refs_with_material,key=lambda _:len(self.refs_with_material[_])):
            print "%s %s" % ( symbol, len(self.refs_with_material[symbol]))
            if len(self.refs_with_material[symbol]) > 20:
                present = self.refs_with_material[symbol][0:10] +  self.refs_with_material[symbol][-10:]
            else:
                present = self.refs_with_material[symbol]
            print "\n".join(map(lambda _:_.id, present))  


class ColladaToChroma(object):
    def __init__(self):
        self.geo = Geometry(detector_material=None)    # bialkali ?
        self.vcount = 0
 
    def visit(self, node):
        self.vcount += 1
        bps = list(node.boundgeom.primitives())
        bpl = bps[0]
        assert len(bps) == 1 and bpl.__class__.__name__ == 'BoundPolylist'
        tris = bpl.triangleset()

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

        if self.vcount % 1000 == 0:
            print node.id
            print self.vcount, bpl, tris, tris.material
            print mesh
            #print mesh.assemble()
            bounds =  mesh.get_bounds()
            extent = bounds[1] - bounds[0]
            print extent



def dump(node):
    print "node.parent\n",node.parent
    print "node nchilden %s nsiblings %s \n%s\n" % (len(node.children),len(node.parent.children),node)






if __name__ == '__main__':
   if len(sys.argv) > 1:
       path = sys.argv[1]
   else:     
       path = '$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae'
   pass    

   DAENode.parse(path)

   #cm = CheckMaterial()
   #DAENode.vwalk(cm.visit)
   #cm.summary()
   
   #cc = ColladaToChroma()
   #DAENode.vwalk(cc.visit)


   for s in DAENode.extra.skinsurface:
       print s 
   for s in DAENode.extra.bordersurface:
       print s 
   for s in DAENode.extra.opticalsurface:
       print s 
   for v,s in DAENode.extra.volmap.items():
       print v
       print s

   print DAENode.extra

   for material in DAENode.orig.materials:
       if material.extra is not None:
           print material
           print material.extra 






if 0:
   DAENode.verbosity = 2 
   for ix in (4000,):
       print ix
       node = DAENode.indexget(ix-1)
       dump(node)











