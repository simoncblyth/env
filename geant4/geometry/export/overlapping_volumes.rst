Fully Overlapping volumes, dodgy dozen
=======================================

distinct volume count discrepancy
---------------------------------

Before including volume name metadata line in *src* name::

    sqlite> select count(distinct(src)) from shape ; 
    12223

After including the the name line, are 6 more distinct::

    simon:export blyth$ echo "select count(distinct(src)) from shape ;" | sqlite3 -noheader g4_00.db 
    12229       


Observe:

#. 6 more after including the volume name comment metadata first line suggests a small number of absolute position duplicated shapes with different volume names
#. confirmed that assertion using `shape.hash` digest that excludes the name metadata 


confirmation of shape overlapping
----------------------------------

Hashing the shape with name excluded confirms issue.
The dodgy dozen, six pairs of volumes are precisely co-located::

    sqlite> select hash, group_concat(name), group_concat(id)  from shape group by hash having count(*) > 1 ;
    hash                              group_concat(name)                                                                                                                           group_concat(id)
    --------------------------------  ---------------------------------------------------------------------------------------------                                                ----------------
    036f14cfb2e7bbe62226d213bd3e7780  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6400,6401       
    2043a400a35f062979ddfa73254cac9d  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6318,6319       
    547dd4e8ad4c711815456951753d8fa9  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4570,4571       
    b7e229d741481e47f3c06236dbc2961d  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  6230,6231       
    be270355bc36384aa290479074aaec4e  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4658,4659       
    c35f0b07cfa25126ec1b156aca3364d8  /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000,/dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000  4740,4741       
    sqlite> 


first degenerate pair
~~~~~~~~~~~~~~~~~~~~~~~

::

    sqlite> select substr(src,0,600) from shape where id = 6401 ;
    #---------- SOLID: /dd/Geometry/CalibrationSources/lvMainSSCavity#pvAmCCo60SourceAcrylic.1000
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
                                            -15954.9 -805788 -4145.32,
                                            -15953.4 -805788 -4145.32,
                                            -15951.7 -805789 -4145.32,
                                            -15950 -805789 -4145.32,
                                            -15948.4 -805788 -4145.32,
                                            -15946.9 -805787 -4145.32,
                                            -15945.8 -805786 -4145.32,
                                            -15945 -805784 -4145.32,
                                            -15944.6 -805783 -4145.32,
                                            -15944.7 -805781 -4145.32,
                                            -15945.3 -8
    sqlite> 
    sqlite> 
    sqlite> 
    sqlite> select substr(src,0,600) from shape where id = 6400 ;
    #---------- SOLID: /dd/Geometry/CalibrationSources/lvMainSSTube#pvMainSSCavity.1000
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
                                            -15954.9 -805788 -4145.32,
                                            -15953.4 -805788 -4145.32,
                                            -15951.7 -805789 -4145.32,
                                            -15950 -805789 -4145.32,
                                            -15948.4 -805788 -4145.32,
                                            -15946.9 -805787 -4145.32,
                                            -15945.8 -805786 -4145.32,
                                            -15945 -805784 -4145.32,
                                            -15944.6 -805783 -4145.32,
                                            -15944.7 -805781 -4145.32,
                                            -15945.3 -805779 -414




