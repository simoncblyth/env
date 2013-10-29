#!/usr/bin/env python
"""

**ABANDONED THIS APPROACH : AS TOO COMPLICATED WRAPPING HEAD AROUND THE 
  PYCOLLADA OBJECT RECURSION, MY OWN VNODE.PY RECURSION IS MORE TRACTABLE**


Monkeypatch pycollada classes
===============================

Attempt to modify pycollada such that BoundGeometry 
instances know where they are in the scene graph.

Need to heed the distinction between the trees

#. xml document tree of xmlnodes
#. Scene Node tree constructed from the xmlnodes, but with considerable 
   re-usage of xmlnodes, such that just copying XML node IDs into 
   scene graph lead to non-unique IDs

Probably the use of `instance_node` NodeNode or `instance_geometry` GeometryNode 
should be the point at which new unique IDs needs to be minted ? 
Probably scene.loadNode ?


This is in to allow access to the material 
of the parent/grandparent nodes.

`BoundGeometry` has `.original` pointing back to the `Geometry` instance

This monkey patch provides a `path` list of node ancestors 
for each `BoundGeometry` instance.

Just reversing back up the tree is not enough, as will not cross 
the containing geometry nodes.  Due to this add an `id` to BoundGeometry
that is the same ad the referencing `NodeNode` (instance_node element).
This allows the corresponding `BoundGeometry` to be looked up 
from ancestor `NodeNode`. To hold this instances a *bound_geometries* 
`IndexedList` is added to the Collada instance. 

Hmm possibly the bound_geometry list should live in the scene 
instance rather than the Collada.

::

    147604     <node id="_dd_Geometry_Sites_lvNearSiteRock0xb5b1f08">
    147605       <instance_geometry url="#near_rock0xb3e6e30">
    147606         <bind_material>
    147607           <technique_common>
    147608             <instance_material symbol="WHITE" target="#_dd_Materials_Rock0x938f188"/>
    147609           </technique_common>
    147610         </bind_material>
    147611       </instance_geometry>
    147612       <node name="_dd_Geometry_Sites_lvNearSiteRock_pvNearHallTop0xb5b1d70">
    147613         <matrix>
    147614                 1 0 0 2500
    147615                 0 1 0 -500
    147616                 0 0 1 7500
    147617                 0.0 0.0 0.0 1.0
    147618         </matrix>
    147619         <instance_node url="#_dd_Geometry_Sites_lvNearHallTop0xb3fa670"/>
    147620       </node>
    147621       <node name="_dd_Geometry_Sites_lvNearSiteRock_pvNearHallBot0xb5b20b0">
    147622         <matrix>
    147623                 1 0 0 0
    147624                 0 1 0 0
    147625                 0 0 1 -5150
    147626                 0.0 0.0 0.0 1.0
    147627         </matrix>
    147628         <instance_node url="#_dd_Geometry_Sites_lvNearHallBot0xb5b1618"/>
    147629       </node>
    147630     </node>

    147631     <node id="World0xb5b2048">
    147632       <instance_geometry url="#WorldBox0xb3e6f60">
    147633         <bind_material>
    147634           <technique_common>
    147635             <instance_material symbol="WHITE" target="#_dd_Materials_Vacuum0x93ab6a0"/>
    147636           </technique_common>
    147637         </bind_material>
    147638       </instance_geometry>
    147639       <node name="_dd_Structure_Sites_db-rock0xb5b2188">
    147640         <matrix>
    147641                 -0.543174 0.83962 0 -16520
    147642                 -0.83962 -0.543174 0 -802110
    147643                  0 0 1 -2110
    147644                  0.0 0.0 0.0 1.0
    147645          </matrix>
    147646         <instance_node url="#_dd_Geometry_Sites_lvNearSiteRock0xb5b1f08"/>
    147647       </node>
    147648     </node>

    147649   </library_nodes>


"""
import collada
from collada.util import IndexedList
original_Collada = collada.Collada

import numpy
import pickle
import os, logging, hashlib
log = logging.getLogger(__name__)

ncount = 0
ocount = 0
rcount = 0

nid = set()
oid = set()

class MonkeyCollada(collada.Collada):
    bound_geometries = property( lambda s: s._bound_geometries, lambda s,v: s._setIndexedList('_bound_geometries', v), doc="""
    A list of :class:`collada.geometry.BoundGeometry` objects. Can also be indexed by id""" )

    def __init__(self, *args, **kwa):
         """
         **MONKEYPATCHED** to add top level matrix applied to all BoundGeometry
         """
         matrix = kwa.pop('matrix', None)
         self._bound_geometries = IndexedList([], ('id',))
         log.info("MonkeyCollada start normal loading %s  " % args )
         original_Collada.__init__(self, *args, **kwa)
         #self._loadBoundGeometries(matrix)

    @classmethod
    def find_unique_id(cls, collection, bid ):
        """
        :param bid: basis id to be extended with `.0` etc..

        Find a unique id for the emerging tree Node, distinct from the source xml node id
        Note that the LV and PV registries are separate
        """
        count = 0 
        uid = None
        while uid is None or uid in collection: 
            uid = "%s.%s" % (bid,count)
            count += 1
        return uid 

    def _loadBoundGeometries(self, matrix=None):
        log.info("_loadBoundGeometries starting") 
        for bg in self.scene.objects('geometry', matrix):
            self.bound_geometries.append(bg) 
        log.info("_loadBoundGeometries loaded %s " % len(self.bound_geometries)) 


class MonkeyBoundGeometry(collada.geometry.BoundGeometry):
    def __str__(self):
        return '<MBoundGeometry %s geom=%s, %d primitives, node depth %d>' % (self.id, self.original.id, len(self), len(self.path))


class MonkeyGeometry(collada.geometry.Geometry):
    def bind(self, matrix, materialnodebysymbol, path=[]):
        """Binds this geometry to a transform matrix and material mapping.
        The geometry's points get transformed by the given matrix and its
        inputs get mapped to the given materials.

        :param numpy.array matrix:
          A 4x4 numpy float matrix
        :param dict materialnodebysymbol:
          A dictionary with the material symbols inside the primitive
          assigned to :class:`collada.scene.MaterialNode` defined in the
          scene

        :rtype: :class:`collada.geometry.BoundGeometry`

        Choosing an id for the BoundGeometry 
        
        * `path[-2].id` copies the LV id (only 249 of those)
        * `path[-3].id` copies the PV id (5643 of those)

        Need to access the containing BoundGeometry 


        """
        bg = MonkeyBoundGeometry(self, matrix, materialnodebysymbol)
        assert path[-2].__class__.__name__ == 'MonkeyNodeNode', "unexpected geometry structure, %s expecting to refer to geomety via an instance_node" % path[-2].__class__.__name__

        bg.path = path
        bg.id = self.collada.find_unique_id( self.collada.bound_geometries, path[-3].id )  # based on the PV id
        return bg 


class MonkeyScene(collada.scene.Scene):
    def objects(self, tipo, matrix=None):
        """Iterate through all objects in the scene that match `tipo`.
        The objects will be bound and transformed via the scene transformations.

        :param str tipo:
          A string for the desired object type. This can be one of 'geometry',
          'camera', 'light', or 'controller'.

        :rtype: generator that yields the type specified

        """
        path = []
        for node in self.nodes:
            for obj in node.objects(tipo, matrix, path=path+[node]): yield obj


class MonkeyGeometryNode(collada.scene.GeometryNode):
    children = property(lambda s:[])
    def visit(self):
        """
        MonkeyGeometryNode count : 12230 distinct instances : 249  
        """
        global ocount
        global oid
        oid.add(id(self))
        ocount += 1
        if ocount % 100 == 0:
            print ocount, self

    def objects(self, tipo, matrix=None, path=[]):
        """Yields a :class:`collada.geometry.BoundGeometry` if ``tipo=='geometry'``"""
        if tipo == 'geometry':
            if matrix is None: matrix = numpy.identity(4, dtype=numpy.float32)
            materialnodesbysymbol = {}
            for mat in self.materials:
                materialnodesbysymbol[mat.symbol] = mat 
            self.visit()
            yield self.geometry.bind(matrix, materialnodesbysymbol, path=path)

    def __str__(self):
       return '<MGeometryNode geometry=%s>' % (self.geometry.id,)

class MonkeyNode(collada.scene.Node):
    def visit_node(self):
        """
        Spins over 24460=12230*2 Nodes alternating between PV and corresponding LV 
        that the PV refers to via an instance_node reference.

        Are these really distinct Node instances or are nodes being recycled ? 
        These are recycled, with only distinct MonkeyNode instances : 5892  

        ::

            1 <MNode top transforms=0, children=1>
            2 <MNode World0xb50dfb8 transforms=0, children=2>
            3 <MNode __dd__Structure__Sites__db-rock0xb50e0f8 transforms=1, children=1>
            4 <MNode __dd__Geometry__Sites__lvNearSiteRock0xb50de78 transforms=0, children=3>
            5 <MNode __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xb50dce0 transforms=1, children=1>
            6 <MNode __dd__Geometry__Sites__lvNearHallTop0xb356a70 transforms=0, children=6>
            7 <MNode __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xb356790 transforms=1, children=1>
            8 <MNode __dd__Geometry__PoolDetails__lvNearTopCover0xb342fe8 transforms=0, children=1>
            9 <MNode __dd__Geometry__Sites__lvNearHallTop--pvNearTeleRpc--pvNearTeleRpc..10xb356ac8 transforms=1, children=1>
            ..... 
            24459 <MNode __dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab90xb50dca8 transforms=1, children=1>
            24460 <MNode __dd__Geometry__RadSlabs__lvNearRadSlab90xb50d530 transforms=0, children=1>


            116814       <node id="__dd__Geometry__Sites__lvNearHallBot--pvNearHallRadSlabs--pvNearHallRadSlab90xb50dca8">
            116815         <matrix>
            116820         </matrix>
            116821         <instance_node url="#__dd__Geometry__RadSlabs__lvNearRadSlab90xb50d530"/>
            116822       </node>

        """
        global ncount
        global nid
        nid.add(id(self))
        ncount += 1
        maxcount = 24460
        if ncount < 10 or ncount > maxcount - 10: 
            print ncount, self    
        if ncount == maxcount:
            print "distinct MonkeyNode instances : %s " % len(nid)

    def recurse(self, node=None, ancestors=[] ):
        """
        This recursively visits 12230*3 = 36690 Nodes.  
        The below pattern of triplets of node types is followed precisely, due to 
        the node/instance_node/instance_geometry layout adopted for the dae file.

        The triplets are collected into VNode on every 3rd leaf node.

        ::

            1 0 <MNode top transforms=0, children=1>
            2 1 <NodeNode node=World0xb50dfb8>
            3 2 <MGeometryNode geometry=WorldBox0xb342f60>

            4 2 <MNode __dd__Structure__Sites__db-rock0xb50e0f8 transforms=1, children=1>
            5 3 <NodeNode node=__dd__Geometry__Sites__lvNearSiteRock0xb50de78>
            6 4 <MGeometryNode geometry=near_rock0xb342e30>

            7 4 <MNode __dd__Geometry__Sites__lvNearSiteRock--pvNearHallTop0xb50dce0 transforms=1, children=1>
            8 5 <NodeNode node=__dd__Geometry__Sites__lvNearHallTop0xb356a70>
            9 6 <MGeometryNode geometry=near_hall_top_dwarf0x92eee48>

            10 6 <MNode __dd__Geometry__Sites__lvNearHallTop--pvNearTopCover0xb356790 transforms=1, children=1>
            11 7 <NodeNode node=__dd__Geometry__PoolDetails__lvNearTopCover0xb342fe8>
            12 8 <MGeometryNode geometry=near_top_cover_box0x92ecf48>

        """
        if node is None:
            node = self

        if len(node.children) == 0:
            VNode.make( ancestors + [node])

        for child in node.children:
            self.recurse(child, ancestors + [node] )


    def objects(self, tipo, matrix=None, path=[]):
        """Iterate through all objects under this node that match `tipo`.
        The objects will be bound and transformed via the scene transformations.

        :param str tipo:
          A string for the desired object type. This can be one of 'geometry',
          'camera', 'light', or 'controller'.
        :param numpy.matrix matrix:
          An optional transformation matrix

        :rtype: generator that yields the type specified

        **MONKEYPATCHED** to record the node path to the geometry
        """
        if matrix != None: M = numpy.dot( matrix, self.matrix )
        else: M = self.matrix
        #self.visit_node() 
        for node in self.children:
            for obj in node.objects(tipo, M, path=path+[node]):
                # summing obj here is mis-leading as every object call will multiply up 
                yield obj

    def __str__(self):
        return '<MNode %s transforms=%d, children=%d>' % (self.id, len(self.transforms), len(self.children))


class MonkeyNodeNode(collada.scene.NodeNode):
    def objects(self, tipo, matrix=None, path=[]):
        """
        NB skip collecting path nodes for NodeNode as it is a reference to another node
        with the same id

        **MONKEYPATCHED** to record the node path to the geometry, for NodeNode traversal simply
        pass the list along to avoid duplicates in the path.
        """
        for obj in self.node.objects(tipo, matrix, path=path):
            yield obj


collada.Collada = MonkeyCollada
collada.geometry.BoundGeometry = MonkeyBoundGeometry
collada.geometry.Geometry = MonkeyGeometry
collada.scene.GeometryNode = MonkeyGeometryNode
collada.scene.Node = MonkeyNode
collada.scene.NodeNode = MonkeyNodeNode
collada.scene.Scene = MonkeyScene


def bound(dae):
    uid = set()
    for i, bg in enumerate(dae.bound_geometries):
        uid.add(bg.id)
        if i % 1000 == 0:
            print i, bg
        pass
    pass    
    assert len(dae.bound_geometries) == len(uid)
    log.info("bound_geometries : %s    distinct id : %s " % (len(dae.bound_geometries),len(uid)) )
    log.info("MonkeyGeometryNode count : %s distint instances : %s  " % (ocount, len(oid)) ) 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = os.path.expandvars("$LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae")
    dae = collada.Collada(path)

    log.info("dae parse completed, now create VNode heirarchy ")
    if os.path.exists(VNode.pkpath):
        VNode.load()
    else:
        top = dae.scene.nodes[0]
        top.recurse()
        VNode.save()

    VNode.walk()


