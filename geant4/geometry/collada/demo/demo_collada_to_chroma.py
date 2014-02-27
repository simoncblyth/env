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
from chroma.geometry import Solid, Geometry, Mesh 

vcount = 0 
nodes_with_material = dict(TOTAL=0)
refs_with_material = dict()


def visit(node):
    global vcount
    vcount += 1

    # counting nodes for each material symbol
    global nodes_with_material
    if not node.symbol in nodes_with_material: 
        nodes_with_material[node.symbol] = 0
    nodes_with_material[node.symbol] += 1
    nodes_with_material['TOTAL'] += 1

    # lists of nodes for each material symbol
    global refs_with_material
    if not node.symbol in refs_with_material:
        refs_with_material[node.symbol] = []
    refs_with_material[node.symbol].append(node)

    if vcount % 1000 == 0:
        print "vcnt    ", vcount
        print "node    ", node
        print "bgeo    ", node.boundgeom
        print "symbol  ", node.symbol
        print "matnode ", node.matnode
        print "matid   ", node.matid


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
   DAENode.vwalk(visit)

   print "vcount %s " % vcount
   print "\n".join(["%-25s : %s " % (symbol,nodes_with_material[symbol]) for symbol in sorted(nodes_with_material,key=lambda _:nodes_with_material[_])]) 

   print "\n".join(["%-25s : %s " % (symbol,len(refs_with_material[symbol])) for symbol in sorted(refs_with_material,key=lambda _:len(refs_with_material[_]))]) 

   for symbol in sorted(refs_with_material,key=lambda _:len(refs_with_material[_])):
       print "%s %s" % ( symbol, len(refs_with_material[symbol]))
       if len(refs_with_material[symbol]) > 20:
           present = refs_with_material[symbol][0:10] +  refs_with_material[symbol][-10:]
       else:
           present = refs_with_material[symbol]
       print "\n".join(map(lambda _:_.id, present))  




if 0:
   DAENode.verbosity = 2 
   for ix in (4000,):
       print ix
       node = DAENode.indexget(ix-1)
       dump(node)











