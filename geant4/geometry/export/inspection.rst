Geometry Inspection
====================

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



