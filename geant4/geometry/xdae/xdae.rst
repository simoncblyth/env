XDAE
=====

Excercising the GDML code based intended DAE exporter.

::

    [blyth@belle7 xdae]$ rm daetest.gdml   # get ABORT if exists
    [blyth@belle7 xdae]$ /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/bin/Linux-g++/xdae
    G4GDML: Reading '/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml' done!
    Stripping off GDML names of materials, solids and volumes ...
    G4DAE: Writing 'daetest.gdml'...
    G4DAE: Writing definitions...
    G4DAE: Writing materials...
    G4DAE: Writing solids...
    G4DAE: Writing structure...
    G4DAE: Writing setup...
    G4DAE: Writing 'daetest.gdml' done !
    [blyth@belle7 xdae]$ 

::

    [blyth@belle7 xdae]$ wc /data1/env/local/env/geant4/geometry/gdml/g4_01.gdml daetest.gdml 
      30946  103505 4111332 /data1/env/local/env/geant4/geometry/gdml/g4_01.gdml
      30882  103185 4100485 daetest.gdml
      61828  206690 8211817 total



pycollada check
-----------------

* http://pycollada.github.io/structure.html

Iterate the nodes finding all bound geometry

* http://pycollada.github.io/reference/generated/collada.scene.Scene.html#collada.scene.Scene.objects

::

    In [34]: boundgeoms = list(dae.scene.objects('geometry'))    # objects method applies transforms

    In [35]: len(boundgeoms)
    Out[35]: 12230

    In [36]: boundgeoms[0]
    Out[36]: <BoundGeometry id=WorldBox0xb3e6f60, 1 primitives>

    In [37]: boundgeoms[-1]
    Out[37]: <BoundGeometry id=near-radslab-box-90xb3e6bd0, 1 primitives>




