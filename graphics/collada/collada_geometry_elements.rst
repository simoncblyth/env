Collada Geometry Elements
==========================

.. contents:: :local:

Objective
----------

Learn about elements of Collada needed to 
represent Geant4 geometry models.

For more Geant4 exporter implementation specifics :doc:`/geant4/geometry/collada/index`


References
-----------

* http://collada.org/mediawiki/index.php/Using_URIs_in_COLLADA




Mapping between Collada model and Geant4 model
------------------------------------------------

================  =============================================
Geant4             Collada
================  =============================================
Solid              instance_geometry OR geometry ?
LogicalVolume      node 
PhysicalVolume     instance_node
================  =============================================



input
-------

*input* declares the input connections to a data source that a consumer requires. A data 
source is a container of raw data that lacks semantic meaning so that the data can be reused within the 
document. To use the data, a consumer declares a connection to it with the desired semantic information. 

::

        <source id="cubeverts-array">
          <float_array count="24" id="cubeverts-array-array">-50 50 50 50 50 50 -50 -50 50 50 -50 50 -50 50 -50 50 50 -50 -50 -50 -50 50 -50 -50</float_array>
          <technique_common>
            <accessor count="8" source="#cubeverts-array-array" stride="3">
              <param type="float" name="X"/>
              <param type="float" name="Y"/>
              <param type="float" name="Z"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="cubeverts-array-vertices">
          <input source="#cubeverts-array" semantic="POSITION"/>
        </vertices>

library_node
--------------

Child elements: One or more nodes.

node
-----

Parent elements: library_nodes, node, visual_scene 
Child elements: 

  * transformational: lookat/matrix/rotate/scale/skew/translate
  * instance_camera/../instance_geometry/instance_node/node
  * extra


The *node* element embodies the hierarchical relationship of elements in a scene
by declaring a point of interest in a scene. 
A node denotes one point on a branch of the scene graph. The *node* element is 
essentially the root of a subgraph of the entire scene graph.  

Within the scene graph abstraction, there are arcs and nodes. 
Nodes are points of information within the graph. 
Arcs connect nodes to other nodes. 
Nodes are further distinguished as interior (branch) nodes and 
exterior (leaf) nodes. 
COLLADA uses the term node to denote interior nodes. Arcs are also called paths. 

The *node* element represents a context in which the child transformation elements are composed in the 
order that they occur. All the other child elements are affected equally by the accumulated transformations 
in the scope of the *node* element. 

The transformation elements transform the coordinate system of the 
*node* element. Mathematically, this means that the transformation elements 
are converted to matrices and postmultiplied in the order in which they are specified within the 
*node* to compose the coordinate system. 


instance_node
----------------------

An *instance_node* creates an instance of an object described by a *node* element. 
Each instance of a *node* element refers to an element in the *node* hierarchy 
that has its own local coordinate system defined for placing objects in the scene. 

Parent elements: *node*

::

    <library_nodes> 
      <node id="myNode"/> 
    </library_nodes> 
    <node> 
      <node> 
        <translate>11.0 12.0 13.0</translate> 
        <instance_node url="#myNode"/> 
      </node> 
    </node> 


Thoughts
~~~~~~~~~~

Looks like Geant4 logical/physical.  But is it beneficial to stay 
at such a high level into the Collada file, perhaps should collapse
to global coordinate space, like VRML2 does. 

If do not do that in the export, then have to do that "flattening" 
within whatever interprets the Collada. Does pycollada do that ? OR is 
pycollada just giving access to the Collada content.

matrix
--------

Parent element: *node*


library_geometries
--------------------

One or more geometry. 

geometry
---------

* only a single convex_mesh, mesh, spline, brep


::
 
  <library_geometries>
    <geometry id="geometry0" name="mycube">
      <mesh>
        <source id="cubenormals-array">
          <float_array count="72" id="cubenormals-array-array">0 0 1 0 0 1 0 0 1 0 0 1 0 1 0 0 1 0 0 1 0 0 1 0 0 -1 0 0 -1 0 0 -1 0 0 -1 0 -1 0 0 -1 0 0 -1 0 0 -1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 -1 0 0 -1 0 0 -1 0 0 -1</float_array>
          <technique_common>
            <accessor count="24" source="#cubenormals-array-array" stride="3">
              <param type="float" name="X"/>
              <param type="float" name="Y"/>
              <param type="float" name="Z"/>
            </accessor>
          </technique_common>
        </source>
        ...
        <triangles count="12" material="materialref">
          <input source="#cubenormals-array" semantic="NORMAL" offset="1"/>
          <input source="#cubeverts-array-vertices" semantic="VERTEX" offset="0"/>
          <p>0 0 2 1 3 2 0 0 3 2 1 3 0 4 1 5 5 6 0 4 5 6 4 7 6 8 7 9 3 10 6 8 3 10 2 11 0 12 4 13 6 14 0 12 6 14 2 15 3 16 7 17 5 18 3 16 5 18 1 19 5 20 7 21 6 22 5 20 6 22 4 23</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
 

mesh
-----

Parent element: *geometry*

Child elements:

* *source* (1+) 
* *vertices* 1
* *lines/linestrips/polygons/polylist/triangles/trifans/tristrips* 0+


The *vertices* element under *mesh* is used to describe mesh-vertices. Polygons, triangles, and so 
forth index mesh-vertices, not positions directly. Mesh-vertices must have at least one *input(unshared)*
element with a semantic attribute whose value is POSITION. 

::

      <mesh>
        <source id="cubenormals-array">
          <float_array count="72" id="cubenormals-array-array">0 0 1 0 0 1 0 0 1 0 0 1 0 1 0 0 1 0 0 1 0 0 1 0 0 -1 0 0 -1 0 0 -1 0 0 -1 0 -1 0 0 -1 0 0 -1 0 0 -1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 0 -1 0 0 -1 0 0 -1 0 0 -1</float_array>
          <technique_common>
            <accessor count="24" source="#cubenormals-array-array" stride="3">
              <param type="float" name="X"/>
              <param type="float" name="Y"/>
              <param type="float" name="Z"/>
            </accessor>
          </technique_common>
        </source>
        <source id="cubeverts-array">
          <float_array count="24" id="cubeverts-array-array">-50 50 50 50 50 50 -50 -50 50 50 -50 50 -50 50 -50 50 50 -50 -50 -50 -50 50 -50 -50</float_array>
          <technique_common>
            <accessor count="8" source="#cubeverts-array-array" stride="3">
              <param type="float" name="X"/>
              <param type="float" name="Y"/>
              <param type="float" name="Z"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="cubeverts-array-vertices">
          <input source="#cubeverts-array" semantic="POSITION"/>
        </vertices>
        <triangles count="12" material="materialref">
          <input source="#cubenormals-array" semantic="NORMAL" offset="1"/>
          <input source="#cubeverts-array-vertices" semantic="VERTEX" offset="0"/>
          <p>0 0 2 1 3 2 0 0 3 2 1 3 0 4 1 5 5 6 0 4 5 6 4 7 6 8 7 9 3 10 6 8 3 10 2 11 0 12 4 13 6 14 0 12 6 14 2 15 3 16 7 17 5 18 3 16 5 18 1 19 5 20 7 21 6 22 5 20 6 22 4 23</p>
        </triangles>
      </mesh>


::

    In [36]: mesh.geometries[0].__class__
    Out[36]: collada.geometry.Geometry

    In [37]: geom = mesh.geometries[0]          

    In [38]: geom.
    geom.bind               geom.createLineSet      geom.createPolylist     geom.double_sided       geom.load               geom.primitives         geom.sourceById         
    geom.collada            geom.createPolygons     geom.createTriangleSet  geom.id                 geom.name               geom.save               geom.xmlnode            

    In [39]: geom.primitives
    Out[39]: [<TriangleSet length=12>]

    In [40]: geom.primitives[0]
    Out[40]: <TriangleSet length=12>

    In [41]: geom.primitives[0][0]
    Out[41]: <Triangle ([-50.  50.  50.], [-50. -50.  50.], [ 50. -50.  50.], "materialref")>

    In [42]: geom.primitives[0][-1]
    Out[42]: <Triangle ([ 50.  50. -50.], [-50. -50. -50.], [-50.  50. -50.], "materialref")>



triangles/trifans/tristrips
---------------------------

For all of them, 

* Each triangle described by the mesh has three vertices. 
* The first triangle is formed from the first, second, and third vertices. 


triangles
    The second triangle is formed from the fourth, fifth, and sixth vertices, and so on. 

trifans
    Each subsequent triangle is formed from the current vertex, reusing the first and the previous vertices. 

tristrips
    Each subsequent triangle is formed from the current vertex, reusing the previous two vertices. 



triangles
-----------

The *p* (stands for **primitive**) element index values indicate the order in which the input values are used.

The indices in a *p* element refer to different inputs depending on their order. 
The first index in a *p* element refers to all inputs with an offset of 0. 
The second index refers to all inputs with an offset of 1. 
Each vertex of the triangle is made up of one index into each input. 
After each input is used, the next index again refers to the inputs with 
offset of 0 and begins a new vertex. 

The winding order of vertices produced is counterclockwise and describes the front side of each triangle. 
If the primitives are assembled without vertex normals then the application may generate per-primitive 
normals to enable lighting. 

::

    <mesh> 
      <source id="position"/> 
      <source id="normal"/> 
      <vertices id="verts"> 
        <input semantic="POSITION" source="#position"/> 
      </vertices> 
      <triangles count="2" material="Bricks"> 
        <input semantic="VERTEX" source="#verts" offset="0"/> 
        <input semantic="NORMAL" source="#normal" offset="1"/> 
        <p> 
          0 0  1 3  2 1   
          0 0  2 1  3 2 
         </p> 
      </triangles> 
    </mesh> 






instance_geometry
-------------------

Child elements: *bind_material* and *extra*

The binding of the geometry to material happens at *instance_geometry* allowing the 
same geometry to be bound to different materials.

The *extra* can provides arbitrary additional information.



::

      <library_visual_scenes>
        <visual_scene id="myscene">
          <node name="node0" id="node0">
            <instance_geometry url="#geometry0">
              <bind_material>
                <technique_common>
                  <instance_material symbol="materialref" target="#material0"/>
                </technique_common>
              </bind_material>
            </instance_geometry>
          </node>
        </visual_scene>
      </library_visual_scenes>


::

    <library_geometries> 
      <geometry id="cube"/> 
    </library_geometries> 
    <node> 
      <node> 
        <translate>11.0 12.0 13.0</translate> 
        <instance_geometry url="#cube"/> 
      </node> 
    </node> 


instance_node
---------------





