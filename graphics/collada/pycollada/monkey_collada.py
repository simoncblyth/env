#!/usr/bin/env python
"""
Monkeypatch pycollada classes
===============================

Attempt to modify pycollada such that BoundGeometry 
instances know where they are in the scene graph.
This is in to allow access to the material 
of the parent/grandparent nodes.


`BoundGeometry` has `.original` pointing back to the `Geometry` instance

This monkey patch 



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
    147615 0 1 0 -500
    147616 0 0 1 7500
    147617 0.0 0.0 0.0 1.0
    147618 </matrix>
    147619         <instance_node url="#_dd_Geometry_Sites_lvNearHallTop0xb3fa670"/>
    147620       </node>
    147621       <node name="_dd_Geometry_Sites_lvNearSiteRock_pvNearHallBot0xb5b20b0">
    147622         <matrix>
    147623                 1 0 0 0
    147624 0 1 0 0
    147625 0 0 1 -5150
    147626 0.0 0.0 0.0 1.0
    147627 </matrix>
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
    147642 -0.83962 -0.543174 0 -802110
    147643 0 0 1 -2110
    147644 0.0 0.0 0.0 1.0
    147645 </matrix>
    147646         <instance_node url="#_dd_Geometry_Sites_lvNearSiteRock0xb5b1f08"/>
    147647       </node>
    147648     </node>
    147649   </library_nodes>






"""
import collada
import numpy
import logging
log = logging.getLogger(__name__)


class MonkeyBoundGeometry(collada.geometry.BoundGeometry):
    pass


class MonkeyGeometry(collada.geometry.Geometry):
    def bind(self, matrix, materialnodebysymbol, geonode=None):
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
        bg.geonode = geonode
        return bg 


class MonkeyGeometryNode(collada.scene.GeometryNode):
    def objects(self, tipo, matrix=None):
        """Yields a :class:`collada.geometry.BoundGeometry` if ``tipo=='geometry'``"""
        if tipo == 'geometry':
            #log.info("monkey GeometryNode") 
            if matrix is None: matrix = numpy.identity(4, dtype=numpy.float32)
            materialnodesbysymbol = {}
            for mat in self.materials:
                materialnodesbysymbol[mat.symbol] = mat 
            yield self.geometry.bind(matrix, materialnodesbysymbol, geonode=self)

class MonkeyScene(collada.scene.Scene):
    def objects(self, tipo):
        """Iterate through all objects in the scene that match `tipo`.
        The objects will be bound and transformed via the scene transformations.

        :param str tipo:
          A string for the desired object type. This can be one of 'geometry',
          'camera', 'light', or 'controller'.

        :rtype: generator that yields the type specified

        """
        matrix = None
        for node in self.nodes:
            for obj in node.objects(tipo, matrix): yield obj


class MonkeyNode(collada.scene.Node):
    def objects(self, tipo, matrix=None):
        """Iterate through all objects under this node that match `tipo`.
        The objects will be bound and transformed via the scene transformations.

        :param str tipo:
          A string for the desired object type. This can be one of 'geometry',
          'camera', 'light', or 'controller'.
        :param numpy.matrix matrix:
          An optional transformation matrix

        :rtype: generator that yields the type specified

        **MONKEYPATCHED** to add parent attribute pointing from child to parent 

        The parent is liable to be already assigned to some other node. So cannot 
        just set it as a node attribute. However a bound node is surely bound 
        to a place in the tree ? So this should be possible.  

        """
        if matrix != None: M = numpy.dot( matrix, self.matrix )
        else: M = self.matrix
        for node in self.children:
            for obj in node.objects(tipo, M):
                yield obj



collada.geometry.BoundGeometry = MonkeyBoundGeometry
collada.geometry.Geometry = MonkeyGeometry
collada.scene.GeometryNode = MonkeyGeometryNode
collada.scene.Node = MonkeyNode
collada.scene.Scene = MonkeyScene




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dae = collada.Collada("test.dae")
    boundgeom = list(dae.scene.objects('geometry'))
    nbg = 0  
    for bg in boundgeom[0:10]:
        nbg += 1
        #if nbg % 100 == 0:
        print bg, bg.geonode, bg.materialnodebysymbol
    print "nbg ", nbg    
       
     


