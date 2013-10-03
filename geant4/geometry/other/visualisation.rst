Geant4 Geometry and Visualization
=====================================

Visualisation, overlap detection are closely related to 
geometry conversion. So tools that do these should be mined for possibilities.

* :google:`geant4 visualization geometry conversion`

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch08.html
* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/Visualization/visdrivers.html
* http://nuclear.unh.edu/~maurik/GEANT4/GEANT4_GEOM.html

Drivers: OpenGL, HepRep, VRML, DAVID, DAWN

* :doc:`/graphics/vrml/vrml` is looking the most promising, as its the most widely supported format exportable


Drivers
--------

::

    [blyth@cms01 visualization]$ du -hs * 
    1.3M    OpenGL
    1.1M    HepRep
    1.1M    management
    884K    OpenInventor
    812K    modeling
    508K    XXX
    428K    VRML
    420K    RayTracer
    380K    externals
    304K    FukuiRenderer
    132K    History
    184K    Tree
    8.0K    GNUmakefile
    [blyth@cms01 visualization]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization


DAVID
-------

* http://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/Detector/geomOverlap.html

The Geant4 DAVID visualization tool can infact automatically detect the
overlaps between the volumes defined in Geant4 and converted to a graphical
representation for visualization purposes. The accuracy of the graphical
representation can be tuned onto the exact geometrical description. In the
debugging, physical-volume surfaces are automatically decomposed into 3D
polygons, and intersections of the generated polygons are investigated. If a
polygon intersects with another one, physical volumes which these polygons
belong to are visualized in color (red is the default).


OpenGL
-------

* http://en.wikipedia.org/wiki/OpenGL

Paint by gl calls approach. Pre-2.0 glsl, so no explicit(useful) geometry conversion.

::

    [blyth@cms01 src]$ vi G4OpenGLSceneHandler.cc
    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/OpenGL/src


