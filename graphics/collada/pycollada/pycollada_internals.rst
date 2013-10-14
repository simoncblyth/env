PyCOLLADA Internals
=====================

Examinine pycollada to see how to extract info needed for Chroma.


XML parsing into IndexLists
------------------------------

Collada object holds IndexLists `_geometries` `_nodes` that behave like both 
lists and dicts based on `id`::

    In [106]: dae._geometries['near-radslab-box-10xb3e60a0']   
    Out[106]: <Geometry id=near-radslab-box-10xb3e60a0, 1 primitives>

    In [118]: len(dae.geometries)   // properties provide dict and list access to these IndexedLists
    Out[118]: 249

    In [125]: len(dae.nodes)
    Out[125]: 249

    In [126]: len(dae.materials)
    Out[126]: 36

    In [119]: dae.geometries['near-radslab-box-10xb3e60a0']
    Out[119]: <Geometry id=near-radslab-box-10xb3e60a0, 1 primitives>

    In [120]: dae.geometries[-1]
    Out[120]: <Geometry id=WorldBox0xb3e6f60, 1 primitives>

    In [121]: dae.geometries[-2]
    Out[121]: <Geometry id=near_rock0xb3e6e30, 1 primitives>

    In [122]: dae.geometries[0]
    Out[122]: <Geometry id=near_top_cover_box0x9390f48, 1 primitives>

    In [123]: dae.geometries[1]
    Out[123]: <Geometry id=RPCStrip0x9391088, 1 primitives>



collada._loadNodes
--------------------

collada/__init__.py::

    397     def _loadNodes(self):
    398         libnodes = self.xmlnode.findall(tag('library_nodes'))
    399         if libnodes is not None:
    400             for libnode in libnodes:
    401                 if libnode is not None:
    402                     tried_loading = []
    403                     succeeded = False
    404                     for node in libnode.findall(tag('node')):
    405                         try:
    406                             N = scene.loadNode(self, node, {})
    407                         except scene.DaeInstanceNotLoadedError as ex:
    408                             tried_loading.append((node, ex))
    409                         except DaeError as ex:
    410                             self.handleError(ex)
    411                         else:
    412                             if N is not None:
    413                                 self.nodes.append(N)
    414                                 succeeded = True


scene.loadNode
~~~~~~~~~~~~~~~~~

Node classes for each XML tag collada/scene.py::

    829 def loadNode( collada, node, localscope ):
    830     """Generic scene node loading from a xml `node` and a `collada` object.
    831 
    832     Knowing the supported nodes, create the appropiate class for the given node
    833     and return it.
    834 
    835     """
    836     if node.tag == tag('node'): return Node.load(collada, node, localscope)
    837     elif node.tag == tag('translate'): return TranslateTransform.load(collada, node)
    838     elif node.tag == tag('rotate'): return RotateTransform.load(collada, node)
    839     elif node.tag == tag('scale'): return ScaleTransform.load(collada, node)
    840     elif node.tag == tag('matrix'): return MatrixTransform.load(collada, node)
    841     elif node.tag == tag('lookat'): return LookAtTransform.load(collada, node)
    842     elif node.tag == tag('instance_geometry'): return GeometryNode.load(collada, node)
    843     elif node.tag == tag('instance_camera'): return CameraNode.load(collada, node)
    844     elif node.tag == tag('instance_light'): return LightNode.load(collada, node)
    845     elif node.tag == tag('instance_controller'): return ControllerNode.load(collada, node)
    846     elif node.tag == tag('instance_node'): return NodeNode.load(collada, node, localscope)
    847     elif node.tag == tag('extra'):
    848         return ExtraNode.load(collada, node)
    849     elif node.tag == tag('asset'):
    850         return None
    851     else: raise DaeUnsupportedError('Unknown scene node %s' % str(node.tag))




Node.load
~~~~~~~~~~~~~

#. transform subnodes are appended to transforms which is used to construct the parent `Node`
   other subnodes become the children

#. source XML id is reused, this *id* will not be unique after instance-ing reuse in scene graph formation

collada/scene.py::

    307 class Node(SceneNode):
    308     """Represents a node object, which is a point on the scene graph, as defined in the collada <node> tag.
    309 
    310     Contains the list of transformations effecting the node as well as any children.
    311     """
    ...
    402     @staticmethod
    403     def load( collada, node, localscope ):
    404         id = node.get('id')
    405         children = []
    406         transforms = []
    407 
    408         for subnode in node:
    409             try:
    410                 n = loadNode(collada, subnode, localscope)
    411                 if isinstance(n, Transform):
    412                     transforms.append(n)
    413                 elif n is not None:
    414                     children.append(n)
    415             except DaeError as ex:
    416                 collada.handleError(ex)
    417 
    418         return Node(id, children, transforms, xmlnode=node)
    419
    420     def __str__(self):
    421         return '<Node transforms=%d, children=%d>' % (len(self.transforms), len(self.children))







Dump source xml::

    In [131]: from collada.xmlutil import etree as ElementTree

    In [136]: print ElementTree.tostring(dae.nodes[-1].xmlnode)

    <node xmlns="http://www.collada.org/2005/11/COLLADASchema" id="World0xb5b2048">
          <instance_geometry url="#WorldBox0xb3e6f60">
            <bind_material>
              <technique_common>
                <instance_material symbol="WHITE" target="#_dd_Materials_Vacuum0x93ab6a0"/>
              </technique_common>
            </bind_material>
          </instance_geometry>
          <node name="_dd_Structure_Sites_db-rock0xb5b2188">
            <matrix>
                                    -0.543174 0.83962 0 -16520
    -0.83962 -0.543174 0 -802110
    0 0 1 -2110
    0.0 0.0 0.0 1.0
    </matrix>
            <instance_node url="#_dd_Geometry_Sites_lvNearSiteRock0xb5b1f08"/>
          </node>
        </node>




Binding objects to a list of transforms
----------------------------------------

::

    029 class Geometry(DaeObject):
    030     """A class containing the data coming from a COLLADA <geometry> tag"""
    031 
    ...
    304     def bind(self, matrix, materialnodebysymbol):
    305         """Binds this geometry to a transform matrix and material mapping.
    306         The geometry's points get transformed by the given matrix and its
    307         inputs get mapped to the given materials.
    308 
    309         :param numpy.array matrix:
    310           A 4x4 numpy float matrix
    311         :param dict materialnodebysymbol:
    312           A dictionary with the material symbols inside the primitive
    313           assigned to :class:`collada.scene.MaterialNode` defined in the
    314           scene
    315 
    316         :rtype: :class:`collada.geometry.BoundGeometry`
    317 
    318         """
    319         return BoundGeometry(self, matrix, materialnodebysymbol)
    320 
    321     def __str__(self):
    322         return '<Geometry id=%s, %d primitives>' % (self.id, len(self.primitives))
    323 
    324     def __repr__(self):
    325         return str(self)
    326 
    327 
    328 class BoundGeometry( object ):
    329     """A geometry bound to a transform matrix and material mapping.
    330         This gets created when a geometry is instantiated in a scene.
    331         Do not create this manually."""
    332 
    333     def __init__(self, geom, matrix, materialnodebysymbol):
    334         self.matrix = matrix
    335         """The matrix bound to"""
    336         self.materialnodebysymbol = materialnodebysymbol
    337         """Dictionary with the material symbols inside the primitive
    338           assigned to :class:`collada.scene.MaterialNode` defined in the
    339           scene"""
    340         self._primitives = geom.primitives
    341         self.original = geom
    342         """The original :class:`collada.geometry.Geometry` object this
    343         is bound to"""
    344 
    345     def __len__(self):
    346         """Returns the number of primitives in the bound geometry"""
    347         return len(self._primitives)
    348 
    349     def primitives(self):
    350         """Returns an iterator that iterates through the primitives in
    351         the bound geometry. Each value returned will be of base type
    352         :class:`collada.primitive.BoundPrimitive`"""
    353         for p in self._primitives:
    354             boundp = p.bind( self.matrix, self.materialnodebysymbol )
    355             yield boundp
    356 
    357     def __str__(self):
    358         return '<BoundGeometry id=%s, %d primitives>' % (self.original.id, len(self))




SceneNodes
~~~~~~~~~~~~
::

     43 class SceneNode(DaeObject):
     44     """Abstract base class for all nodes within a scene."""
     45 
     46     def objects(self, tipo, matrix=None):
     47         """Iterate through all objects under this node that match `tipo`.
     48         The objects will be bound and transformed via the scene transformations.
     49 
     50         :param str tipo:
     51           A string for the desired object type. This can be one of 'geometry',
     52           'camera', 'light', or 'controller'.
     53         :param numpy.matrix matrix:
     54           An optional transformation matrix
     55 
     56         :rtype: generator that yields the type specified
     57 
     58         """
     59         pass

::

    simon:collada blyth$ grep SceneNode *.py
    scene.py:class SceneNode(DaeObject):
    scene.py:class Node(SceneNode):
    scene.py:class GeometryNode(SceneNode):
    scene.py:class ControllerNode(SceneNode):
    scene.py:class MaterialNode(SceneNode):
    scene.py:class CameraNode(SceneNode):
    scene.py:class LightNode(SceneNode):
    scene.py:class ExtraNode(SceneNode):



::

    479 class GeometryNode(SceneNode):
    480     """Represents a geometry instance in a scene, as defined in the collada <instance_geometry> tag."""
    481 
    ...   
    515     def objects(self, tipo, matrix=None):
    516         """Yields a :class:`collada.geometry.BoundGeometry` if ``tipo=='geometry'``"""
    517         if tipo == 'geometry':
    518             if matrix is None: matrix = numpy.identity(4, dtype=numpy.float32)
    519             materialnodesbysymbol = {}
    520             for mat in self.materials:
    521                 materialnodesbysymbol[mat.symbol] = mat
    522             yield self.geometry.bind(matrix, materialnodesbysymbol)


