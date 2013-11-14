WRL Crosscheck : reveals mangled vertex coordinates
======================================================

Initial issue identified below, pursued in :ref:`vertex_coordinate_debug`

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


