Overview
===========

.. contents:: :local:

Objective
---------

Convert a Geant4 geometry into a surface based STL file format for use on GPUs.

* :google:`Convert a Geant4 geometry into a surface based STL file format`
* :google:`ST-Viewer step tools GDML`

Requirements
--------------

* reduce to the simplicity of a bunch of triangles
* two material indices, inside+outside for every tri
* sensitive detectors, handled via a special material index
* optical surface properties, presumably can be handled via material index
  
* if 2 materials/colors per tri not supported 

  * copy and flip faces to provide a place to attach the 2nd material ?
  * pack both materials in the place of one ? 
  * does this have to get collapsed down to 16 bits in STL later anyhow ?

* originating volume identity, probably not needed (but useful for development)


Subfolder purposes
-----------------------------

DAEFILE
    Extraction of relevant parts of VRML2FILE visualization from Geant4, enabling creation of libDAEFILE.so
xdaefile
    Geant4 main executable xdaefile that uses libDAEFILE.so to read a GDML file and export WRL (and maybe DAE) 
DAE
    Extraction of relevant parts of GDML persistency writing from Geant4, enabling creation of libDAE.so. Starting from 
    `cp -R /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/persistency/gdml DAE`
xdae
    Use of libDAE.so


Geometry Representations
-------------------------

detdesc
        source xml
geant4
        in memory tree of PV+LV+Solid instances 
gdml
        xml persistence, with model close to geant4
vrml2
        visualization targetted, a list of shapes each composed of triangle/quad faces and vertices
x3d
        later incarnation of vrml2 
stl
        triangles only, normal + 3 vertices + UINT16 per-tri color
        http://en.wikipedia.org/wiki/STL_(file_format)
collada
        widely supported (including OSX SceneKit, Xcode, Preview), flexible
        http://en.wikipedia.org/wiki/COLLADA
        http://collada.org/


Commercial Converters
----------------------

* http://www.solveering.com/instep.htm includes GDML support/conversions



