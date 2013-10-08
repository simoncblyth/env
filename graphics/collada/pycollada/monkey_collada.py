#!/usr/bin/env python
"""
Monkeypatch pycollada classes
===============================

Attempt to modify pycollada such that BoundGeometry 
instances know where they are in the scene graph.
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
import logging
log = logging.getLogger(__name__)


class MonkeyCollada(collada.Collada):
    bound_geometries = property( lambda s: s._bound_geometries, lambda s,v: s._setIndexedList('_bound_geometries', v), doc="""
    A list of :class:`collada.geometry.BoundGeometry` objects. Can also be indexed by id""" )

    def __init__(self, *args, **kwa):
         """
         **MONKEYPATCHED** to add top level matrix applied to all BoundGeometry
         """
         matrix = kwa.pop('matrix')
         self._bound_geometries = IndexedList([], ('id',))
         log.info("MonkeyCollada start normal loading ")
         original_Collada.__init__(self, *args, **kwa)
         self._loadBoundGeometries(matrix)

    def _loadBoundGeometries(self, matrix=None):
        log.info("_loadBoundGeometries starting") 
        for bg in self.scene.objects('geometry', matrix):
            self.bound_geometries.append(bg) 
        log.info("_loadBoundGeometries loaded %s " % len(self.bound_geometries)) 


class MonkeyBoundGeometry(collada.geometry.BoundGeometry):
    def __str__(self):
        return '<MBoundGeometry id=%s geom=%s, %d primitives, node depth %d>' % (self.id, self.original.id, len(self), len(self.path))


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

        """
        bg = MonkeyBoundGeometry(self, matrix, materialnodebysymbol)
        bg.path = path
        assert path[-2].__class__.__name__ == 'MonkeyNodeNode', "unexpected geometry structure, %s expecting to refer to geomety via an instance_node" % path[-2].__class__.__name__
        bg.id = path[-2] # formerly path[-2].id suffering stomping   
        ## setting MonkeyBoundGeometry id the same as the MonkeyNodeNode which referred to it, ie the instance_node pluck 
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
    def objects(self, tipo, matrix=None, path=[]):
        """Yields a :class:`collada.geometry.BoundGeometry` if ``tipo=='geometry'``"""
        if tipo == 'geometry':
            #log.info("monkey GeometryNode") 
            if matrix is None: matrix = numpy.identity(4, dtype=numpy.float32)
            materialnodesbysymbol = {}
            for mat in self.materials:
                materialnodesbysymbol[mat.symbol] = mat 
            yield self.geometry.bind(matrix, materialnodesbysymbol, path=path)


class MonkeyNode(collada.scene.Node):
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
        for node in self.children:
            for obj in node.objects(tipo, M, path=path+[node]):
                yield obj

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dae = collada.Collada("test.dae")
    for i, bg in enumerate(dae.bound_geometries):
        print i, bg, bg.materialnodebysymbol

       
     


