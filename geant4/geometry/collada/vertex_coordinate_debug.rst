
.. _vertex_coordinate_debug:

Vertex Coordinate Debug : finding the invertex rotation matrix
================================================================

.. contents:: :local:

Coordinate Systems
-------------------

Collada
~~~~~~~~~

* :google:`Collada Coordinate system`
* http://collada.org/public_forum/showthread.php/828-Coordinate-System-clarification

There is an *UP_AXIS* control in *asset* elements which defaults to *Y_UP*

Geant4
~~~~~~~~

* :google:`geant4 coordinate system`

  * http://hypernews.slac.stanford.edu/HyperNews/geant4/get/visualization/521.html  left handed system OR not, apparently not 

    * 2008 : vis looks to one user to be left-handed,   +y up, +x right, -z towards viewpoint (not +z toward viewpoint as documented below)
    * 2011 : fobs off as optical illusion 

  * http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch08s04.html

By default, the up-Vector is parallel to the y-axis and the viewpoint direction
is parallel to the z-axis, so the the view shows the x-axis to the right and
the y-axis upwards - a projection on to the canonical x-y plane 

GDML
~~~~~

Coordinates system, same as Geant4 presumably.


VRML2
~~~~~~~

* :google:`VRML2 Coordinate system`

  * http://www.mathworks.com/help/sl3d/vrml.html

::

    VRML2 coordinate system

       +Y
        |
        | 
        |
        +-------- +X 
       /
      /
     /
    +Z


The VRML coordinate system is different from the MATLAB and Aerospace Blockset
coordinate systems. VRML uses the world coordinate system in which the y-axis
points upward and the z-axis places objects nearer or farther from the front of
the screen. It is important to realize this fact in situations involving the
interaction of these different coordinate systems. SimMechanics uses the same
coordinate system as VRML.

Rotation angles - In VRML, rotation angles are defined using the right-hand
rule. Imagine your right hand holding an axis while your thumb points in the
direction of the axis toward its positive end. Your four remaining fingers
point in a counterclockwise direction. This counterclockwise direction is the
positive rotation angle of an object moving around that axis.



GDML Geometry Trace
---------------------

N:/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml::

    2162     <box lunit="mm" name="near_rock_main0xb752940" x="50000" y="50000" z="50000"/>     # Z -25000 : +25000 
    2163     <box lunit="mm" name="near_rock_void0xb82e300" x="50010" y="50010" z="12010"/>     # Z  -6005 : +6005   (but shunted down so)    -25005 : -12995
    2164     <subtraction name="near_rock0xb8499c8">
    2165       <first ref="near_rock_main0xb752940"/>
    2166       <second ref="near_rock_void0xb82e300"/>                                    # subtracting off the bigger box : is this to avoid numerical issues ??? 
    2167       <position name="near_rock0xb8499c8_pos" unit="mm" x="0" y="0" z="-19000"/>        ## (50-12)/2=19   half dim in Z
    2168     </subtraction>
    2169     <box lunit="mm" name="WorldBox0xc6328f0" x="4800000" y="4800000" z="4800000"/>
    2170   </solids>


* http://lcgapp.cern.ch/project/simu/framework/GDML/doc/GDMLmanual.pdf

The GDML Boolean Solids can be described using following Boolean operations: union, 
subtraction and intersection. As for Geant4 Boolean operations, the second solid is placed 
with given position and rotation in the system coordinates of the first solid. 

::

    30919     <volume name="/dd/Geometry/Sites/lvNearSiteRock0xb82e578">
    30920       <materialref ref="/dd/Materials/Rock0xb849090"/>
    30921       <solidref ref="near_rock0xb8499c8"/>
    30922       <physvol name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xb7dd068">
    30923         <volumeref ref="/dd/Geometry/Sites/lvNearHallTop0xb745f10"/>
    30924         <position name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xb7dd068_pos" unit="mm" x="2500" y="-500" z="7500"/>
    30925       </physvol>
    30926       <physvol name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xc5065d0">
    30927         <volumeref ref="/dd/Geometry/Sites/lvNearHallBot0xb7dd4a8"/>
    30928         <position name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xc5065d0_pos" unit="mm" x="0" y="0" z="-5150"/>
    30929       </physvol>
    30930     </volume>
    30931     <volume name="World0xc6337a8">
    30932       <materialref ref="/dd/Materials/Vacuum0xbaff828"/>
    30933       <solidref ref="WorldBox0xc6328f0"/>                 ##   -2400k +2400k box
    30934       <physvol name="/dd/Structure/Sites/db-rock0xc633af8">
    30935         <volumeref ref="/dd/Geometry/Sites/lvNearSiteRock0xb82e578"/>
    30936         <position name="/dd/Structure/Sites/db-rock0xc633af8_pos" unit="mm" x="-16519.9999999999" y="-802110" z="-2110"/>
    30937         <rotation name="/dd/Structure/Sites/db-rock0xc633af8_rot" unit="deg" x="0" y="0" z="-122.9"/>
    30938       </physvol>
    30939     </volume>
    30940   </structure>



Debug PV1 `near_rock` vertices
---------------------------------


pycollada Access
~~~~~~~~~~~~~~~~~

::

    In [220]: import lxml.etree as ET

    In [6]: dae = collada.Collada("0.dae")

    In [7]: top = dae.scene.nodes[0]

    In [8]: top
    Out[8]: <Node transforms=0, children=1>

    In [11]: boundgeom = list(top.objects("geometry"))

    In [12]: len(boundgeom)
    Out[12]: 12230


PV0 WorldBox
~~~~~~~~~~~~~~


::

    In [151]: boundgeom[0]
    Out[151]: <BoundGeometry id=WorldBox0xa8bff60, 1 primitives>

    In [162]: for po in list(boundgeom[0].primitives())[0]:print po, po.indices
    <Polygon vertices=4> [0 3 2 1]
    <Polygon vertices=4> [4 7 3 0]
    <Polygon vertices=4> [7 6 2 3]
    <Polygon vertices=4> [6 5 1 2]
    <Polygon vertices=4> [5 4 0 1]
    <Polygon vertices=4> [4 5 6 7]

    In [163]: boundgeom[0].original.primitives[0].vertex
    Out[163]: 
    array([[-2400000., -2400000., -2400000.],...    ## actually dimensions of boundgeom[0] the worldbox not relevant, just provides the frame



PV1 near_rock untransformed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [179]: boundgeom[1]
    Out[179]: <BoundGeometry id=near_rock0xa8bfe30, 1 primitives>

    In [194]: boundgeom[1].original.primitives[0].vertex     # more relevant, this is what gets transformed
    Out[194]: 
    array([[-25000.        , -25000.        ,  25000.        ],
           [ 25000.        , -25000.        ,  25000.        ],
           [ 25000.        ,  25000.        ,  25000.        ],
           [-25000.        ,  25000.        ,  25000.        ],
           [-25000.        ,  25000.        , -12993.79980469],
           [-25000.        , -25000.        , -12993.79980469],
           [ 25000.        ,  25000.        , -12993.79980469],
           [ 25000.        , -25000.        , -12993.79980469]], dtype=float32)


    In [221]: print ET.tostring(boundgeom[1].original.xmlnode)
    <geometry xmlns="http://www.collada.org/2005/11/COLLADASchema" id="near_rock0xa8bfe30" name="near_rock0xa8bfe30">
          <mesh>
            <source id="near_rock0xa8bfe30-Pos">
              <float_array count="24" id="near_rock0xa8bfe30-Pos-array">
                                    -25000 -25000 25000 
                                    25000 -25000 25000 
                                    25000 25000 25000 
                                    -25000 25000 25000 
                                    -25000 25000 -12993.8 
                                    -25000 -25000 -12993.8 
                                    25000 25000 -12993.8 
                                    25000 -25000 -12993.8 
    </float_array>


PV1 near_rock transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    In [24]: top.children[0].node.children[1].id
    Out[24]: '__dd__Structure__Sites__db-rock0xaa8b0f8'

    In [23]: print ET.tostring(top.children[0].node.children[1].transforms[0].xmlnode)
    <matrix xmlns="http://www.collada.org/2005/11/COLLADASchema">
          -0.543174 0.83962 0 -16520
          -0.83962 -0.543174 0 -802110
           0 0 1 -2110
           0.0 0.0 0.0 1.0
    </matrix>

    In [177]: boundgeom[1].matrix[:3,3]
    Out[177]: array([ -16520., -802110.,   -2110.], dtype=float32)           # expected translation from GDML

            
    In [315]: collada.scene.makeRotationMatrix(0,0,1,-numpy.pi*122.9/180.)    # -122.9 deg is from the GDML
    Out[315]: 
    array([[-0.54317445,  0.83961987,  0.        ,  0.        ],
           [-0.83961987, -0.54317445,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=float32)

    In [178]: boundgeom[1].matrix[:3,:3]                                     # rotation anti-clockwise about z axis by -122.9 degrees
    Out[178]: 
    array([[-0.54317403,  0.83961999,  0.        ],
           [-0.83961999, -0.54317403,  0.        ],
           [ 0.        ,  0.        ,  1.        ]], dtype=float32)

    In [183]: math.cos(-122.9*math.pi/180.)
    Out[183]: -0.54317444995067088

    In [184]: math.sin(-122.9*math.pi/180.)
    Out[184]: -0.83961986453441306

::

      cos th   -sin th    0     # th rotation anti-clockwise about z axis 
      sin th    cos th    0
        0         0       1


PV1 primitives
~~~~~~~~~~~~~~~~

::

    In [197]: for po in boundgeom[1].original.primitives[0]:print po, po.indices
    <Polygon vertices=4> [0 1 2 3]
    <Polygon vertices=3> [4 5 0]
    <Polygon vertices=3> [0 3 4]
    <Polygon vertices=3> [6 4 3]
    <Polygon vertices=3> [3 2 6]
    <Polygon vertices=3> [7 6 2]
    <Polygon vertices=3> [2 1 7]
    <Polygon vertices=3> [5 7 1]
    <Polygon vertices=3> [1 0 5]
    <Polygon vertices=3> [5 4 6]
    <Polygon vertices=3> [6 7 5]



Unsuccessfull diddling
~~~~~~~~~~~~~~~~~~~~~~~


Trying to rotate/reflect things around failed to achieve a PV1 match.

::

    In [30]: zrot_ = lambda _:numpy.asmatrix(numpy.array( [[math.cos(_), -math.sin(_), 0],[math.sin(_), math.cos(_), 0],[0, 0, 1]] ))

    In [31]: list(boundgeom[1].primitives())[0].vertex * zrot_(math.pi/2.)
    Out[31]: 
    matrix([[-767540.125     ,   23931.1484375 ,   22890.        ],
            [-809521.125     ,   51089.8515625 ,   22890.        ],
            [-836679.875     ,    9108.85058594,   22890.        ],
            [-794698.875     ,  -18049.8515625 ,   22890.        ],
            [-794698.875     ,  -18049.8515625 ,  -15103.79980469],
            [-767540.125     ,   23931.1484375 ,  -15103.79980469],
            [-836679.875     ,    9108.85058594,  -15103.79980469],
            [-809521.125     ,   51089.8515625 ,  -15103.79980469]])

    In [32]: list(boundgeom[1].primitives())[0].vertex * zrot_(-math.pi/2.)
    Out[32]: 
    matrix([[ 767540.125     ,  -23931.1484375 ,   22890.        ],
            [ 809521.125     ,  -51089.8515625 ,   22890.        ],
            [ 836679.875     ,   -9108.85058594,   22890.        ],
            [ 794698.875     ,   18049.8515625 ,   22890.        ],
            [ 794698.875     ,   18049.8515625 ,  -15103.79980469],
            [ 767540.125     ,  -23931.1484375 ,  -15103.79980469],
            [ 836679.875     ,   -9108.85058594,  -15103.79980469],
            [ 809521.125     ,  -51089.8515625 ,  -15103.79980469]])


    In [35]: xyref = numpy.asmatrix(numpy.array([[0,1,0],[1,0,0],[0,0,1]]))

    In [36]: xyref
    Out[36]: 
    matrix([[0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]])

    In [37]: list(boundgeom[1].primitives())[0].vertex * xyref
    Out[37]: 
    matrix([[-767540.125     ,  -23931.1484375 ,   22890.        ],
            [-809521.125     ,  -51089.8515625 ,   22890.        ],
            [-836679.875     ,   -9108.85058594,   22890.        ],
            [-794698.875     ,   18049.8515625 ,   22890.        ],
            [-794698.875     ,   18049.8515625 ,  -15103.79980469],
            [-767540.125     ,  -23931.1484375 ,  -15103.79980469],
            [-836679.875     ,   -9108.85058594,  -15103.79980469],
            [-809521.125     ,  -51089.8515625 ,  -15103.79980469]])




reconcile PV1 vertices, VRML2 cf pycollada
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [269]: C0 = boundgeom[1].original.primitives[0].vertex    # collada vertices before transformation

    In [270]: C0
    Out[270]: 
    array([[-25000.        , -25000.        ,  25000.        ],
           [ 25000.        , -25000.        ,  25000.        ],
           [ 25000.        ,  25000.        ,  25000.        ],
           [-25000.        ,  25000.        ,  25000.        ],
           [-25000.        ,  25000.        , -12993.79980469],
           [-25000.        , -25000.        , -12993.79980469],
           [ 25000.        ,  25000.        , -12993.79980469],
           [ 25000.        , -25000.        , -12993.79980469]], dtype=float32)

    In [266]: M = numpy.asmatrix(boundgeom[1].matrix).transpose()

    In [276]: ( M[:3,:3] * C0.T ).T           # transposed collada vertices *pre*-multiplied by the rotation matrix (no translation)
    Out[276]:                                 #   EUREKA : THIS MATCHES THE VRML2 COORDINATES : "V" below
    matrix([[ 34569.8515625 ,  -7411.14941406,  25000.        ],
            [  7411.14941406,  34569.8515625 ,  25000.        ],
            [-34569.8515625 ,   7411.14941406,  25000.        ],
            [ -7411.14941406, -34569.8515625 ,  25000.        ],
            [ -7411.14941406, -34569.8515625 , -12993.79980469],
            [ 34569.8515625 ,  -7411.14941406, -12993.79980469],
            [-34569.8515625 ,   7411.14941406, -12993.79980469],
            [  7411.14941406,  34569.8515625 , -12993.79980469]], dtype=float32)


    In [363]: numpy.dot( C0, boundgeom[1].matrix[:3,:3] )         ## simpler way to do the above avoiding the transposing and keeping post-multiplication
    Out[363]: 
    array([[ 34569.8515625 ,  -7411.14941406,  25000.        ],
           [  7411.14941406,  34569.8515625 ,  25000.        ],
           [-34569.8515625 ,   7411.14941406,  25000.        ],
           [ -7411.14941406, -34569.8515625 ,  25000.        ],
           [ -7411.14941406, -34569.8515625 , -12993.79980469],
           [ 34569.8515625 ,  -7411.14941406, -12993.79980469],
           [-34569.8515625 ,   7411.14941406, -12993.79980469],
           [  7411.14941406,  34569.8515625 , -12993.79980469]], dtype=float32)



    In [287]: C0 * M[:3,:3]                   # post multiplication (as done by pycollada) leads to vertices that look vaguely similar, with maybe an xy swap, 
                                              # but failed to find a rotation + reflection to line them up 
                                              # .... the problem was the transpose done to the rotation matrix , sign flip issue
    Out[287]: 
    matrix([[ -7411.14941406,  34569.8515625 ,  25000.        ],
            [-34569.8515625 ,  -7411.14941406,  25000.        ],
            [  7411.14941406, -34569.8515625 ,  25000.        ],
            [ 34569.8515625 ,   7411.14941406,  25000.        ],
            [ 34569.8515625 ,   7411.14941406, -12993.79980469],
            [ -7411.14941406,  34569.8515625 , -12993.79980469],
            [  7411.14941406, -34569.8515625 , -12993.79980469],
            [-34569.8515625 ,  -7411.14941406, -12993.79980469]], dtype=float32)


PV1 VRML2 coordinates
~~~~~~~~~~~~~~~~~~~~~~~

::

    sqlite> select sid,id, x,y,z from wrl.point where sid=1 ;
    sid         id          x                y           z         
    ----------  ----------  ---------------  ----------  ----------
    1           0           18049.900390625  -809521.0   22890.0   
    1           1           -9108.860351562  -767540.0   22890.0   
    1           2           -51089.8984375   -794699.0   22890.0   
    1           3           -23931.09960937  -836680.0   22890.0   
    1           4           -23931.09960937  -836680.0   -15104.200
    1           5           18049.900390625  -809521.0   -15104.200
    1           6           -51089.8984375   -794699.0   -15104.200
    1           7           -9108.860351562  -767540.0   -15104.200


::

    simon:~ blyth$ shapedb-shape 1
    #---------- SOLID: /dd/Structure/Sites/db-rock.1000
            Shape {
                    appearance Appearance {
                            material Material {
                                    diffuseColor 1 1 1
                                    transparency 0.7
                            }
                    }
                    geometry IndexedFaceSet {
                            coord Coordinate {
                                    point [
                                            18049.9 -809521 22890,
                                            -9108.86 -767540 22890,
                                            -51089.9 -794699 22890,
                                            -23931.1 -836680 22890,
                                            -23931.1 -836680 -15104.2,
                                            18049.9 -809521 -15104.2,
                                            -51089.9 -794699 -15104.2,
                                            -9108.86 -767540 -15104.2,
                                    ]
                            }
                            coordIndex [
                                    0, 1, 2, 3, -1,
                                    4, 5, 0, -1,
                                    0, 3, 4, -1,
                                    6, 4, 3, -1,
                                    3, 2, 6, -1,
                                    7, 6, 2, -1,
                                    2, 1, 7, -1,
                                    5, 7, 1, -1,
                                    1, 0, 5, -1,
                                    5, 4, 6, -1,
                                    6, 7, 5, -1,
                            ]
                            solid FALSE
                    }
            }


::

    In [104]: from env.geant4.geometry.vrml2.vrml2db import VRML2DB

    In [105]: db = VRML2DB()

    In [286]: a = db.points(1) ; a            # VRML2 points from the shape db  
    Out[286]: 
    array([[  18049.90039062, -809521.        ,   22890.        ],
           [  -9108.86035156, -767540.        ,   22890.        ],
           [ -51089.8984375 , -794699.        ,   22890.        ],
           [ -23931.09960938, -836680.        ,   22890.        ],
           [ -23931.09960938, -836680.        ,  -15104.20019531],
           [  18049.90039062, -809521.        ,  -15104.20019531],
           [ -51089.8984375 , -794699.        ,  -15104.20019531],
           [  -9108.86035156, -767540.        ,  -15104.20019531]], dtype=float32)


    In [285]: V = a - boundgeom[1].matrix[:3,3] ; V    # VRML2 points with translation taken out    
    Out[285]: 
    array([[ 34569.8984375 ,  -7411.        ,  25000.        ],
           [  7411.13964844,  34570.        ,  25000.        ],
           [-34569.8984375 ,   7411.        ,  25000.        ],
           [ -7411.09960938, -34570.        ,  25000.        ],
           [ -7411.09960938, -34570.        , -12994.20019531],
           [ 34569.8984375 ,  -7411.        , -12994.20019531],
           [-34569.8984375 ,   7411.        , -12994.20019531],
           [  7411.13964844,  34570.        , -12994.20019531]], dtype=float32)



Transformation Implementations
--------------------------------

PyCollada Transformation Inconsistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyCollada transformations, hmm inconsistent pre/post-multiplication a bug somewhere::

    simon:collada blyth$ grep ":3,:3" *.py
    light.py:        self.position = numpy.dot( matrix[:3,:3], plight.position ) + matrix[:3,3]
    light.py:        self.direction = numpy.dot( matrix[:3,:3], dlight.direction )
    lineset.py:            self._vertex = numpy.asarray(ls._vertex * M[:3,:3]) + matrix[:3,3]
    lineset.py:            self._normal = numpy.asarray(ls._normal * M[:3,:3])
    polylist.py:        self._vertex = None if pl._vertex is None else numpy.asarray(pl._vertex * M[:3,:3]) + matrix[:3,3]
    polylist.py:        self._normal = None if pl._normal is None else numpy.asarray(pl._normal * M[:3,:3])
    triangleset.py:        self._vertex = None if ts.vertex is None else numpy.asarray(ts._vertex * M[:3,:3]) + matrix[:3,3]
    triangleset.py:        self._normal = None if ts._normal is None else numpy.asarray(ts._normal * M[:3,:3])


PyCollada Primitive Fix
~~~~~~~~~~~~~~~~~~~~~~~~~~

Testing primfix in daegeom.py shows that switching to using non-transposed rotation matrix and 
keeping post-multiplication achieves a match for PV1

::

    simon:~ blyth$ daegeom.py $LOCAL_BASE/env/graphics/collada/0.dae 1
    INFO:env.graphics.collada.pycollada.daegeom:dump_geom from /usr/local/env/graphics/collada/0.dae boundgeom index 1 
    before primfix <BoundPolylist length=11> nvtx: 8
    [[ -23931.1484375  -767540.125        22890.        ]
     [ -51089.8515625  -809521.125        22890.        ]
     [  -9108.85058594 -836679.875        22890.        ]
     [  18049.8515625  -794698.875        22890.        ]
     [  18049.8515625  -794698.875       -15103.79980469]
     [ -23931.1484375  -767540.125       -15103.79980469]
     [  -9108.85058594 -836679.875       -15103.79980469]
     [ -51089.8515625  -809521.125       -15103.79980469]]
    after primfix <BoundPolylist length=11> nvtx: 8
    [[  18049.8515625  -809521.125        22890.        ]
     [  -9108.85058594 -767540.125        22890.        ]
     [ -51089.8515625  -794698.875        22890.        ]
     [ -23931.1484375  -836679.875        22890.        ]
     [ -23931.1484375  -836679.875       -15103.79980469]
     [  18049.8515625  -809521.125       -15103.79980469]
     [ -51089.8515625  -794698.875       -15103.79980469]
     [  -9108.85058594 -767540.125       -15103.79980469]]
    from VRML2DB: 
    [[  18049.90039062 -809521.           22890.        ]
     [  -9108.86035156 -767540.           22890.        ]
     [ -51089.8984375  -794699.           22890.        ]
     [ -23931.09960938 -836680.           22890.        ]
     [ -23931.09960938 -836680.          -15104.20019531]
     [  18049.90039062 -809521.          -15104.20019531]
     [ -51089.8984375  -794699.          -15104.20019531]
     [  -9108.86035156 -767540.          -15104.20019531]]

::

    sqlite> select sid,id,x,y,z from dae.point where sid=1 ;
    sid         id          x               y            z         
    ----------  ----------  --------------  -----------  ----------
    1           0           -23931.1484375  -767540.125  22890.0   
    1           1           -51089.8515625  -809521.125  22890.0   
    1           2           -9108.85058593  -836679.875  22890.0   
    1           3           18049.8515625   -794698.875  22890.0   
    1           4           18049.8515625   -794698.875  -15104.200
    1           5           -23931.1484375  -767540.125  -15104.200
    1           6           -9108.85058593  -836679.875  -15104.200
    1           7           -51089.8515625  -809521.125  -15104.200



G4DAEWriteStructure::MatrixWrite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     42 void G4DAEWriteStructure::MatrixWrite(xercesc::DOMElement* nodeElement, const G4Transform3D& T)
     43 {
     44     std::ostringstream ss ;
     45     // row-major order 
     46 
     47     ss << "\n\t\t\t\t" ;
     48     ss << T.xx() << " " ;
     49     ss << T.xy() << " " ;
     50     ss << T.xz() << " " ;
     51     ss << T.dx() << "\n" ;
     52 
     53     ss << T.yx() << " " ;
     54     ss << T.yy() << " " ;
     55     ss << T.yz() << " " ;
     56     ss << T.dy() << "\n" ;
     57 
     58     ss << T.zx() << " " ;
     59     ss << T.zy() << " " ;
     60     ss << T.zz() << " " ;
     61     ss << T.dz() << "\n" ;
     62 
     63     ss << "0.0 0.0 0.0 1.0\n" ;
     64 
     65     std::string fourbyfour = ss.str();
     66     xercesc::DOMElement* matrixElement = NewTextElement("matrix", fourbyfour);
     67     nodeElement->appendChild(matrixElement);
     68 }

::

     71 void G4DAEWriteStructure::PhysvolWrite(xercesc::DOMElement* parentNodeElement,
     72                                         const G4VPhysicalVolume* const physvol,
     73                                         const G4Transform3D& T,
     74                                         const G4String& ModuleName)
     75 {
     76    const G4String pvname = GenerateName(physvol->GetName(),physvol);
     77    const G4String lvname = GenerateName(physvol->GetLogicalVolume()->GetName(),physvol->GetLogicalVolume() );
     78 
     79    G4int copyNo = physvol->GetCopyNo();  //why always zero ?
     80    if(copyNo != 0) G4cout << "G4DAEWriteStructure::PhysvolWrite " << pvname << " " << copyNo << G4endl ;
     81 
     82    xercesc::DOMElement* childNodeElement = NewElementOneNCNameAtt("node","id",pvname);
     83    MatrixWrite( childNodeElement, T );
     84 
     85    xercesc::DOMElement* instanceNodeElement = NewElementOneNCNameAtt("instance_node", "url", lvname , true);
     86 
     87    childNodeElement->appendChild(instanceNodeElement);
     88    parentNodeElement->appendChild(childNodeElement);
     89 }

::

    145 G4Transform3D G4DAEWriteStructure::
    146 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    147 {
    148    if (VolumeMap().find(volumePtr) != VolumeMap().end())
    149    {
    150        return VolumeMap()[volumePtr]; // Volume is already processed
    151    }
    152 
    153    G4VSolid* solidPtr = volumePtr->GetSolid();
    154    G4Transform3D R,invR;
    ...
    175    const G4int daughterCount = volumePtr->GetNoDaughters();
    ...
    180    for (G4int i=0;i<daughterCount;i++)   // Traverse all the children!
    181    {
    182       const G4VPhysicalVolume* const physvol = volumePtr->GetDaughter(i);
    ...
    185       G4Transform3D daughterR;
    187       daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);
    188 
    189       G4RotationMatrix rot;
    190       if (physvol->GetFrameRotation() != 0)
    191       {
    192          rot = *(physvol->GetFrameRotation());
    193       }
    194       G4Transform3D P(rot,physvol->GetObjectTranslation());
    195       PhysvolWrite(nodeElement,physvol,invR*P*daughterR,ModuleName);
    196    }
    ...
    199    structureElement->appendChild(nodeElement);  
    ...  // appended after  traversing children
    203 
    204    VolumeMap()[volumePtr] = R;
    ...  
    210    return R;
    211 }


Need to debug thus, 

#. looks like `R,invR,daughterR` will all always be identity matrices, 
#. makes the `P` transform blissfully PV local, this is kinda what is needed  


G4GDMLReadStructure::PhysvolRead  rotation inverted ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. hmm the PhysvolRead inverts the rotation 

::

    256 void G4GDMLReadStructure::
    257 PhysvolRead(const xercesc::DOMElement* const physvolElement)
    258 {
    ...
    318    G4Transform3D transform(GetRotationMatrix(rotation).inverse(),position);
    319    transform = transform*G4Scale3D(scale.x(),scale.y(),scale.z());
    320 
    321    G4String pv_name = logvol->GetName() + "_PV";
    322    G4PhysicalVolumesPair pair = G4ReflectionFactory::Instance()
    323      ->Place(transform,pv_name,logvol,pMotherLogical,false,0,check);
    324 
    325    if (pair.first != 0) { GeneratePhysvolName(name,pair.first); }
    326    if (pair.second != 0) { GeneratePhysvolName(name,pair.second); }


G4Transform3D::G4Transform3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

source/global/HEPGeometry/include/G4Transform3D.hh::

     34 #include <CLHEP/Geometry/Transform3D.h>
     35 
     36 typedef HepGeom::Transform3D G4Transform3D;
     37 
     38 typedef HepGeom::Rotate3D G4Rotate3D;
     39 typedef HepGeom::RotateX3D G4RotateX3D;
     40 typedef HepGeom::RotateY3D G4RotateY3D;
     41 typedef HepGeom::RotateZ3D G4RotateZ3D;
     42 
     43 typedef HepGeom::Translate3D G4Translate3D;
     44 typedef HepGeom::TranslateX3D G4TranslateX3D;
     45 typedef HepGeom::TranslateY3D G4TranslateY3D;
     46 typedef HepGeom::TranslateZ3D G4TranslateZ3D;


* /data1/env/local/dyb/external/build/LCG/clhep/2.0.4.2/CLHEP/Geometry/Geometry/Transform3D.h
* /data1/env/local/dyb/external/build/LCG/clhep/2.0.4.2/CLHEP/Geometry/Geometry/Transform3D.icc


PyCollada Recursive Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within a node the matrix is composed from `I*t[0]*t[1]` for G4DAEWrite a single matrix is written only
so no complications here. However moving to decomposed R and T might be beneficial.

::

    307 class Node(SceneNode):
    308     """Represents a node object, which is a point on the scene graph, as defined in the collada <node> tag.
    309 
    310     Contains the list of transformations effecting the node as well as any children.
    311     """
    312 
    313     def __init__(self, id, children=None, transforms=None, xmlnode=None):
    ...
    335         self.transforms = []
    336         if transforms is not None:
    337             self.transforms = transforms
    338         """A list of transformations effecting the node. This can
    339           contain any object that inherits from :class:`collada.scene.Transform`"""
    340         self.matrix = numpy.identity(4, dtype=numpy.float32)
    341         """A numpy.array of size 4x4 containing a transformation matrix that
    342         combines all the transformations in :attr:`transforms`. This will only
    343         be updated after calling :meth:`save`."""
    344 
    345         for t in self.transforms:
    346             self.matrix = numpy.dot(self.matrix, t.matrix)
    ...
    358     def objects(self, tipo, matrix=None):
    359         """Iterate through all objects under this node that match `tipo`.
    360         The objects will be bound and transformed via the scene transformations.
    361 
    362         :param str tipo:
    363           A string for the desired object type. This can be one of 'geometry',
    364           'camera', 'light', or 'controller'.
    365         :param numpy.matrix matrix:
    366           An optional transformation matrix
    367 
    368         :rtype: generator that yields the type specified
    369 
    370         """
    371         if matrix != None: M = numpy.dot( matrix, self.matrix )
    372         else: M = self.matrix
    373         for node in self.children:
    374             for obj in node.objects(tipo, M):
    375                 yield obj



Current recursion level matrix `self.matrix` post-multiplies the the matrix passed from parent, 
so where the `matrix` from above is in brackets you end up with::

    ( PV0 ) * PV1        
    ( PV0 * PV1 ) * PV2       
    ( PV0 * PV1 * PV2 ) * PV3     



 
Geant4 Transform handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc::

    413 void G4VRML2SCENEHANDLER::BeginPrimitives(const G4Transform3D& objectTransformation)
    414 {
    415   G4VSceneHandler::BeginPrimitives (objectTransformation);
    416   fpObjectTransformation = &objectTransformation;
    417 #if defined DEBUG_SCENE_FUNC
    418     G4cerr << "***** BeginPrimitives " << "\n" ;
    419 #endif
    420     VRMLBeginModeling();
    421 }

geant4.9.2.p01/source/visualization/management/src/G4VisManager.cc::

     447 void G4VisManager::Draw (const G4Polyhedron& polyhedron,
     448              const G4Transform3D& objectTransform) {
     449   if (IsValidView ()) {
     450     ClearTransientStoreIfMarked();
     451     fpSceneHandler -> BeginPrimitives (objectTransform);
     452     fpSceneHandler -> AddPrimitive (polyhedron);
     453     fpSceneHandler -> EndPrimitives ();
     454   }
     455 }


Geant4 using recursive post-multiplication in G4PhysicalVolumeModel::DescribeAndDescend::

    336 void G4PhysicalVolumeModel::DescribeAndDescend
    337 (G4VPhysicalVolume* pVPV,
    338  G4int requestedDepth,
    339  G4LogicalVolume* pLV,
    340  G4VSolid* pSol,
    341  G4Material* pMaterial,
    342  const G4Transform3D& theAT,
    343  G4VGraphicsScene& sceneHandler)
    344 {
    345   // Maintain useful data members...
    346   fpCurrentPV = pVPV;
    347   fpCurrentLV = pLV;
    348   fpCurrentMaterial = pMaterial;
    349 
    350   const G4RotationMatrix objectRotation = pVPV -> GetObjectRotationValue ();
    351   const G4ThreeVector&  translation     = pVPV -> GetTranslation ();
    352   G4Transform3D theLT (G4Transform3D (objectRotation, translation));
    353 
    354   // Compute the accumulated transformation...
    355   // Note that top volume's transformation relative to the world
    356   // coordinate system is specified in theAT == startingTransformation
    357   // = fTransform (see DescribeYourselfTo), so first time through the
    358   // volume's own transformation, which is only relative to its
    359   // mother, i.e., not relative to the world coordinate system, should
    360   // not be accumulated.
    361   G4Transform3D theNewAT (theAT);
    362   if (fCurrentDepth != 0) theNewAT = theAT * theLT;
    363   fpCurrentTransform = &theNewAT;
    364 



