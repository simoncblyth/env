Geometry Inspection
====================

.. contents:: :local:

Create Geometry DB from .wrl
-----------------------------

::

    cd ~/e/geant4/geometry/export
    ./vrml2file.py --help
    ./vrml2file.py -c g4_00.wrl     


Open Geometry DB
------------------

::

    simon:export blyth$ sqlite3 g4_00.db
    -- Loading resources from /Users/blyth/.sqliterc

    SQLite version 3.7.14.1 2012-10-04 19:37:12
    Enter ".help" for instructions
    Enter SQL statements terminated with a ";"
    sqlite> 


Ordering by decreasing extent in x 
----------------------------------

::

    sqlite> select dx, dy, dz, name from xshape join shape on xshape.sid == shape.id order by dx desc limit 100 ;
    dx          dy          dz          name                                                                                                
    ----------  ----------  ----------  ---------------------------------------------------------------------------------------------       
    69139.8     69140.0     37994.2     /dd/Structure/Sites/db-rock.1000                                                                    
    36494.56    45091.0     15000.29    /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000                                                
    20239.92    21988.0     293.51      /dd/Geometry/Sites/lvNearHallTop#pvNearRPCRoof.1003                                                 
    19770.52    21602.0     294.0       /dd/Geometry/Sites/lvNearHallTop#pvNearRPCSptRoof.1004                                              
    17916.63    19696.0     10300.0     /dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot.1001                                                
    13823.3     15602.0     10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead.1000                                                
    13823.18    15602.0     300.0       /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9.1009                         
    13823.07    15602.0     44.0        /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000                                       
    ...
    4759.2      7326.0      40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearLongEdgeUnistruts:2#pvNearLongQuadEdgeUnistrus
    4759.2      7326.0      40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearLongEdgeUnistruts:2#pvNearLongQuadEdgeUnistrus
    4759.2      7326.0      40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearLongEdgeUnistruts:2#pvNearLongQuadEdgeUnistrus
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvTopReflector.1429                                                           
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvBotReflector.1430                                                           
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvTopReflector.1429                                                           
    4494.3      4495.0      20.0        /dd/Geometry/AD/lvOIL#pvBotReflector.1430                                                           
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap.1000                                              
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap.1000                                              
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap.1000                                              
    4444.3      4445.0      0.20000000  /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap.1000                                              
    4440.3      4441.0      0.10000000  /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR.1000                                                    
    4440.3      4441.0      0.09999999  /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR.1000                                                    
    4440.3      4441.0      0.10000000  /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR.1000                                                    
    4440.3      4441.0      0.09999999  /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR.1000                                                    
    4217.5      1184.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8.1008                         
    4217.5      1184.0      10000.0     /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab4.1004                         
    4074.8      4075.0      4094.71     /dd/Geometry/AD/lvOIL#pvOAV.1000                                                                    
    4074.8      4075.0      4094.71     /dd/Geometry/AD/lvOIL#pvOAV.1000                                                                    
    3958.9      3959.0      4076.53     /dd/Geometry/AD/lvOAV#pvLSO.1000                                                                    
    3958.9      3959.0      4076.53     /dd/Geometry/AD/lvOAV#pvLSO.1000                                                                    
    3919.6      878.0       40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:1#pvNearQuadCornerUnistrus:2#pvC
    3919.6      878.0       40.0        /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:1#pvNearQuadCornerUnistrus:2#pvC


Squeeze
-------

::

    sqlite> select point.id, (x-ax)/dx, (y-ay)/dy, (z-az)/dz, dx, dy, dz from point join xshape join shape on point.sid = xshape.sid and point.sid = shape.id where point.sid=11663 ;
    id          (x-ax)/dx   (y-ay)/dy   (z-az)/dz   dx          dy          dz                                                                                                  
    ----------  ----------  ----------  ----------  ----------  ----------  ---------------------------------------------------------------------------------------------       
    0           -0.4900722  0.11818181  -0.5        55.3999999  55.0        2860.0                                                                                              
    1           -0.0965703  -0.5        -0.5        55.3999999  55.0        2860.0                                                                                              
    2           0.50992779  -0.1        -0.5        55.3999999  55.0        2860.0                                                                                              
    3           0.11642599  0.5         -0.5        55.3999999  55.0        2860.0                                                                                              
    4           -0.4900722  0.11818181  0.5         55.3999999  55.0        2860.0                                                                                              
    5           -0.0965703  -0.5        0.5         55.3999999  55.0        2860.0                                                                                              
    6           0.50992779  -0.1        0.5         55.3999999  55.0        2860.0                                                                                              
    7           0.11642599  0.5         0.5         55.3999999  55.0        2860.0                                                                                              
    8           0.10198555  0.44545454  -0.5        55.3999999  55.0        2860.0                                                                                              
    9           -0.4648014  0.08181818  -0.5        55.3999999  55.0        2860.0                                                                                              
    10          0.44494584  -0.0818181  -0.5        55.3999999  55.0        2860.0                                                                                              
    11          -0.1218411  -0.4636363  -0.5        55.3999999  55.0        2860.0                                                                                              
    12          -0.4648014  0.08181818  0.5         55.3999999  55.0        2860.0                                                                                              
    13          -0.1218411  -0.4636363  0.5         55.3999999  55.0        2860.0                                                                                              
    14          0.10198555  0.44545454  0.5         55.3999999  55.0        2860.0                                                                                              
    15          0.44494584  -0.0818181  0.5         55.3999999  55.0        2860.0       





Inspect Shapes heads
----------------------

Heads of all shapes are identical::

    sqlite> select distinct(substr(src,0,178)) from shape ;
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


::

    simon:export blyth$ echo select src from shape where id=12222 \; | sqlite3 -noheader g4_00.db 
    #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab2.1002
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
                                            -22540.9 -796477 -12260,
                                            -22834.2 -796414 -12260,
                                            -23724.9 -800569 -12260,
                                            -23431.5 -800632 -12260,
                                            -22540.9 -796477 -2260,
                                            -22834.2 -796414 -2260,
                                            -23724.9 -800569 -2260,
                                            -23431.5 -800632 -2260,
                                    ]
                            }
                            coordIndex [
                                    0, 3, 2, 1, -1,
                                    4, 7, 3, 0, -1,
                                    7, 6, 2, 3, -1,
                                    6, 5, 1, 2, -1,
                                    5, 4, 0, 1, -1,
                                    4, 5, 6, 7, -1,
                            ]
                            solid FALSE
                    }
            }




Small number of different src lengths
----------------------------------------

Only ~53 different lengths of src but 12k distinct src. 
Small number of shapes are repeated in different positions, eg PMT rotations.

::

    sqlite> select len,count(*) as N from shape group by len order by len ;
    31|5362
    36|1
    45|163
    47|160
    52|1
    ...
    859|672
    892|6
    941|64
    961|2
    979|672
    1031|2
    1291|672
    1588|6
    1707|2
    1869|2


