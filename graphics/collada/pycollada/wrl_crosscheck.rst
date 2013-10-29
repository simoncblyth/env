WRL Crosscheck
================

::

    simon:collada blyth$ t shapedb-shape
    shapedb-shape is a function
    shapedb-shape () 
    { 
        echo select src from shape where id=${1:-0} limit 1 \; | sqlite3 -noheader -list $(shapedb-path)
    }

    simon:collada blyth$ shapedb-shape 3199
    #---------- SOLID: /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt.1
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
                                            -16657.8 -801370 -8842.5,
                                            -16654.9 -801373 -8808.59,
                                            -16648.2 -801368 -8809.75,
                                            -16642 -801362 -8813.14,
                                            -16636.7 -801358 -8818.53,


Name is not unique in shape table but the *id* which corresponds to recursion index is::

    sqlite> select id, name from shape where name like '/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt.%' ;
    id          name                                                                                                                                                  
    ----------  ---------------------------------------------------------------------------------------------                                                         
    3199        /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt.1                                       
    4859        /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt.1                                       
    sqlite> 


3199 corresponds to `__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0.dae`

::

    simon:collada blyth$ grep __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8 vnodetree.txt
                                                 [11.1] VNode(23,25)[3199,__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0] __dd__Materials__Pyrex0x8885198  
                                                 [11.1] VNode(23,25)[4859,__dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.1] __dd__Materials__Pyrex0x8885198  


Hmm the coordinates are world ones, so need to access the bound geometry from the full geometry.


* http://localhost:8080/dump/3199?ancestors=1&children=1
* http://localhost:8080/dump/3199?geometry=1

Hmm resemblance but no match

::

    _dump [3199] => ids [3199] 
    cfg {'geometry': u'1'} 

    VNode(23,25)[3199]    __dd__Geometry__AD__lvOIL--pvAdPmtArray--pvAdPmtArrayRotated--pvAdPmtRingInCyl..1--pvAdPmtInRing..1--pvAdPmtUnit--pvAdPmt0xa8d92d8.0             __dd__Materials__Pyrex0x8885198 
    <BoundGeometry id=pmt-hemi0x88414e8, 1 primitives>
    [[ -17153.55273438 -802530.75         -8842.5       ]
     [ -17147.41210938 -802529.375        -8808.88476562]
     [ -17154.84375    -802524.5625       -8808.88476562]
     ..., 
     [ -17056.34765625 -802410.25         -8803.47167969]
     [ -17056.35742188 -802410.25         -8803.46875   ]
     [ -17065.29296875 -802404.5          -8800.6171875 ]]



::

    sqlite> select id,x,y,z from point where sid=3199 ; 
    id          x           y           z         
    ----------  ----------  ----------  ----------
    0           -16657.8    -801370.0   -8842.5   
    1           -16654.9    -801373.0   -8808.59  
    2           -16648.2    -801368.0   -8809.75  
    3           -16642.0    -801362.0   -8813.14  
    ...
    357         -16580.8    -801506.0   -8812.63  
    358         -16574.1    -801501.0   -8805.92  
    359         -16574.1    -801501.0   -8805.91  
    360         -16566.3    -801494.0   -8801.7   
    361         -16566.3    -801494.0   -8801.69  
    sqlite> 



Its not from the other AD::

    sqlite> select id,x,y,z from point where sid=4859 ;
    id          x           y           z         
    ----------  ----------  ----------  ----------
    0           -13538.9    -806191.0   -8842.5   
    1           -13536.0    -806194.0   -8808.59  
    2           -13529.3    -806189.0   -8809.75  
    3           -13523.1    -806183.0   -8813.14  
    4           -13517.7    -806179.0   -8818.53  
    ...
    359         -13455.2    -806322.0   -8805.91  
    360         -13447.4    -806315.0   -8801.7   
    361         -13447.4    -806315.0   -8801.69  
    sqlite> 



Volume 1 (0 not in WRL), hmm coordinate swappings ? maybe Y up effect ?

* http://localhost:8080/dump/1?geometry=1


::

    _dump [1] => ids [1] 
    cfg {'geometry': u'1'} 

    VNode(3,5)[1]    __dd__Structure__Sites__db-rock0xaa8b0f8.0             __dd__Materials__Rock0x8868188 
    <BoundGeometry id=near_rock0xa8bfe30, 1 primitives>
    nvtx:8
    [[ -23931.1484375  -767540.125        22890.        ]
     [ -51089.8515625  -809521.125        22890.        ]
     [  -9108.85058594 -836679.875        22890.        ]
     [  18049.8515625  -794698.875        22890.        ]
     [  18049.8515625  -794698.875       -15103.79980469]
     [ -23931.1484375  -767540.125       -15103.79980469]
     [  -9108.85058594 -836679.875       -15103.79980469]
     [ -51089.8515625  -809521.125       -15103.79980469]]


                                            -9108.86 -767540 22890,
                                            18049.9 -809521 22890,
                                            -23931.1 -836680 22890,
                                            -51089.9 -794699 22890,

                                            -51089.9 -794699 -15104.2,
                                            -9108.86 -767540 -15104.2,
                                            -23931.1 -836680 -15104.2,
                                            18049.9 -809521 -15104.2,
 
All the same X, Y and Z numbers (with some precision difference) are there, BUT with swapped x-y pairings between points ?


::

    simon:collada blyth$ shapedb-shape 1   
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

    dae = collada.Collada("0.dae")
    top = dae.scene.nodes[0]
    boundgeom = list(top.objects('geometry'))
    len(boundgeom)   # 12230

    In [70]: bg = boundgeom[1]

    In [73]: bpl = list(bg.primitives())[0]    # always? one primitive BoundPolyList 

    In [76]: for po in bpl.polygons():print po, po.indices
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

    In [79]: bpl.vertex
    Out[79]: 
    array([[ -23931.1484375 , -767540.125     ,   22890.        ],
           [ -51089.8515625 , -809521.125     ,   22890.        ],
           [  -9108.85058594, -836679.875     ,   22890.        ],
           [  18049.8515625 , -794698.875     ,   22890.        ],
           [  18049.8515625 , -794698.875     ,  -15103.79980469],
           [ -23931.1484375 , -767540.125     ,  -15103.79980469],
           [  -9108.85058594, -836679.875     ,  -15103.79980469],
           [ -51089.8515625 , -809521.125     ,  -15103.79980469]], dtype=float32)



    In [98]: pl =  bg.original.primitives[0]

    In [99]: pl.nindices
    Out[99]: 2

    In [100]: pl.vcounts
    Out[100]: array([4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    In [101]: pl.vertex
    Out[101]: 
    array([[-25000.        , -25000.        ,  25000.        ],
           [ 25000.        , -25000.        ,  25000.        ],
           [ 25000.        ,  25000.        ,  25000.        ],
           [-25000.        ,  25000.        ,  25000.        ],
           [-25000.        ,  25000.        , -12993.79980469],
           [-25000.        , -25000.        , -12993.79980469],
           [ 25000.        ,  25000.        , -12993.79980469],
           [ 25000.        , -25000.        , -12993.79980469]], dtype=float32)


    In [131]: pl._vertex.shape 
    Out[131]: (8, 3)

    In [133]: matrix.shape
    Out[133]: (4, 4)

    In [135]: M[:3,:3].shape
    Out[135]: (3, 3)


    In [129]: numpy.asarray(pl._vertex * M[:3,:3]) + matrix[:3,3]
    Out[129]: 
    array([[ -23931.1484375 , -767540.125     ,   22890.        ],
           [ -51089.8515625 , -809521.125     ,   22890.        ],
           [  -9108.85058594, -836679.875     ,   22890.        ],
           [  18049.8515625 , -794698.875     ,   22890.        ],
           [  18049.8515625 , -794698.875     ,  -15103.79980469],
           [ -23931.1484375 , -767540.125     ,  -15103.79980469],
           [  -9108.85058594, -836679.875     ,  -15103.79980469],
           [ -51089.8515625 , -809521.125     ,  -15103.79980469]], dtype=float32)

    In [145]: matrix[:3,3]
    Out[145]: array([ -16520., -802110.,   -2110.], dtype=float32)




Matrix handling from  /usr/local/env/graphics/collada/pycollada/collada/polylist.py::

    302 class BoundPolylist(primitive.BoundPrimitive):
    303     """A polylist bound to a transform matrix and materials mapping.
    304 
    305     * If ``P`` is an instance of :class:`collada.polylist.BoundPolylist`, then ``len(P)``
    306       returns the number of polygons in the set. ``P[i]`` returns the i\ :sup:`th`
    307       polygon in the set.
    308     """
    309 
    310     def __init__(self, pl, matrix, materialnodebysymbol):
    311         """Create a bound polylist from a polylist, transform and material mapping.
    312         This gets created when a polylist is instantiated in a scene. Do not create this manually."""
    313         M = numpy.asmatrix(matrix).transpose()
    314         self._vertex = None if pl._vertex is None else numpy.asarray(pl._vertex * M[:3,:3]) + matrix[:3,3]
    315         self._normal = None if pl._normal is None else numpy.asarray(pl._normal * M[:3,:3])
    316         self._texcoordset = pl._texcoordset
    317         matnode = materialnodebysymbol.get( pl.material )
    318         if matnode:
    319             self.material = matnode.target
    320             self.inputmap = dict([ (sem, (input_sem, set)) for sem, input_sem, set in matnode.inputs ])
    321         else: self.inputmap = self.material = None
    322         self.index = pl.index
    323         self.nvertices = pl.nvertices
    324         self._vertex_index = pl._vertex_index
    325         self._normal_index = pl._normal_index
    326         self._texcoord_indexset = pl._texcoord_indexset
    327         self.polyindex = pl.polyindex
    328         self.npolygons = pl.npolygons
    329         self.matrix = matrix
    330         self.materialnodebysymbol = materialnodebysymbol
    331         self.original = pl



::

    In [93]: import lxml.etree as ET

    In [94]: print ET.tostring(bg.original.xmlnode)
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
              <technique_common>
                <accessor count="8" source="#near_rock0xa8bfe30-Pos-array" stride="3">
                  <param name="X" type="float"/>
                  <param name="Y" type="float"/>
                  <param name="Z" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <source id="near_rock0xa8bfe30-Norm">
              <float_array count="33" id="near_rock0xa8bfe30-Norm-array">
                                    0 -0 1 
                                    -1 0 0 
                                    -1 -0 -0 
                                    0 1 -0 
                                    0 1 0 
                                    1 0 -0 
                                    1 -0 0 
                                    0 -1 0 
                                    0 -1 -0 
                                    0 0 -1 
                                    -0 0 -1 
    </float_array>
              <technique_common>
                <accessor count="11" source="#near_rock0xa8bfe30-Norm-array" stride="3">
                  <param name="X" type="float"/>
                  <param name="Y" type="float"/>
                  <param name="Z" type="float"/>
                </accessor>
              </technique_common>
            </source>
            <vertices id="near_rock0xa8bfe30-Vtx">
              <input semantic="POSITION" source="#near_rock0xa8bfe30-Pos"/>
            </vertices>
            <polylist count="11" material="WHITE">
              <input offset="0" semantic="VERTEX" source="#near_rock0xa8bfe30-Vtx"/>
              <input offset="1" semantic="NORMAL" source="#near_rock0xa8bfe30-Norm"/>
              <vcount>4 3 3 3 3 3 3 3 3 3 3 </vcount>
              <p>
                 0 0  1 0  2 0  3 0  
                 4 1  5 1  0 1   
                 0 2  3 2  4 2   
                 6 3  4 3  3 3   
                 3 4  2 4  6 4   
                 7 5  6 5  2 5   
                 2 6  1 6  7 6   
                 5 7  7 7  1 7   
                 1 8  0 8  5 8   
                 5 9  4 9  6 9   
                 6 10  7 10  5 10  
               </p>
            </polylist>
          </mesh>
        </geometry>


::

    In [76]: for po in bpl.polygons():print po, po.indices
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





Examine world box according to pycollada
-----------------------------------------


::

    In [42]: dae = collada.Collada("0.dae")
    In [43]: top = dae.scene.nodes[0]
    In [44]: boundgeom = list(top.objects('geometry'))
    In [45]: len(boundgeom)
    Out[45]: 12230

    In [46]: boundgeom[0]
    Out[46]: <BoundGeometry id=WorldBox0xa8bff60, 1 primitives>

    In [47]: bg = boundgeom[0]

    In [48]: bg.matrix
    Out[48]: 
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]], dtype=float32)

    In [51]: list(bg.primitives())[0]
    Out[51]: <BoundPolylist length=6>

    In [52]: bpl = list(bg.primitives())[0]

    In [53]: bpl.
    bpl.index                 bpl.material              bpl.matrix                bpl.normal_index          bpl.nvertices             bpl.polygons              bpl.shapes                bpl.texcoordset           bpl.vertex                
    bpl.inputmap              bpl.materialnodebysymbol  bpl.normal                bpl.npolygons             bpl.original              bpl.polyindex             bpl.texcoord_indexset     bpl.triangleset           bpl.vertex_index          

    In [53]: bpl.npolygons   # 6 faces 
    Out[53]: 6

    In [54]: bpl.nvertices   # 4 * 6  : repeating the vertices for each face
    Out[54]: 24


    In [57]: for po in bpl.polygons():print po
    <Polygon vertices=4>
    <Polygon vertices=4>
    <Polygon vertices=4>
    <Polygon vertices=4>
    <Polygon vertices=4>
    <Polygon vertices=4>

    In [60]: for po in bpl.polygons():print po.indices, po.vertices
    [0 3 2 1] [[-2400000. -2400000. -2400000.]
     [-2400000.  2400000. -2400000.]
     [ 2400000.  2400000. -2400000.]
     [ 2400000. -2400000. -2400000.]]
    [4 7 3 0] [[-2400000. -2400000.  2400000.]
     [-2400000.  2400000.  2400000.]
     [-2400000.  2400000. -2400000.]
     [-2400000. -2400000. -2400000.]]
    [7 6 2 3] [[-2400000.  2400000.  2400000.]
     [ 2400000.  2400000.  2400000.]
     [ 2400000.  2400000. -2400000.]
     [-2400000.  2400000. -2400000.]]
    [6 5 1 2] [[ 2400000.  2400000.  2400000.]
     [ 2400000. -2400000.  2400000.]
     [ 2400000. -2400000. -2400000.]
     [ 2400000.  2400000. -2400000.]]
    [5 4 0 1] [[ 2400000. -2400000.  2400000.]
     [-2400000. -2400000.  2400000.]
     [-2400000. -2400000. -2400000.]
     [ 2400000. -2400000. -2400000.]]
    [4 5 6 7] [[-2400000. -2400000.  2400000.]
     [ 2400000. -2400000.  2400000.]
     [ 2400000.  2400000.  2400000.]
     [-2400000.  2400000.  2400000.]]

    In [61]: print bpl.vertex
    [[-2400000. -2400000. -2400000.]
     [ 2400000. -2400000. -2400000.]
     [ 2400000.  2400000. -2400000.]
     [-2400000.  2400000. -2400000.]
     [-2400000. -2400000.  2400000.]
     [ 2400000. -2400000.  2400000.]
     [ 2400000.  2400000.  2400000.]
     [-2400000.  2400000.  2400000.]]

    In [62]: bpl.
    bpl.index                 bpl.material              bpl.matrix                bpl.normal_index          bpl.nvertices             bpl.polygons              bpl.shapes                bpl.texcoordset           bpl.vertex                
    bpl.inputmap              bpl.materialnodebysymbol  bpl.normal                bpl.npolygons             bpl.original              bpl.polyindex             bpl.texcoord_indexset     bpl.triangleset           bpl.vertex_index          

    In [62]: bpl.vertex_index
    Out[62]: 
    array([0, 3, 2, 1, 4, 7, 3, 0, 7, 6, 2, 3, 6, 5, 1, 2, 5, 4, 0, 1, 4, 5, 6,
           7])

    In [63]: len(bpl.vertex_index)
    Out[63]: 24





