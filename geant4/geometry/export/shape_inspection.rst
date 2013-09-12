Geometry Shape Inspection
==========================

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


