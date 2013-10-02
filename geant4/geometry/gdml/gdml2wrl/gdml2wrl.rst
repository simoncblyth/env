GDML2WRL
==========

Building with cmake
--------------------

#. note that the root source dir of a project contains a `CMakeLists.txt` file, subdirs also contain these
#. create a build directory, that can be within the source dir or elsewhere
#. invoke cmake with argument pointing back to the source dir

::

    [blyth@belle7 gdml2wrl]$ mkdir -p /tmp/env/gdml2wrl.build
    [blyth@belle7 gdml2wrl]$ pwd
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl
    [blyth@belle7 gdml2wrl]$ cd /tmp/env/gdml2wrl.build
    [blyth@belle7 gdml2wrl.build]$ cmake /home/blyth/e/geant4/geometry/gdml/gdml2wrl
    -- The C compiler identification is GNU
    -- The CXX compiler identification is GNU
    -- Check for working C compiler: /usr/bin/gcc
    -- Check for working C compiler: /usr/bin/gcc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /tmp/env/gdml2wrl.build
    [blyth@belle7 gdml2wrl.build]$ 
    [blyth@belle7 gdml2wrl.build]$ make
    Scanning dependencies of target gdml2wrl
    [100%] Building CXX object CMakeFiles/gdml2wrl.dir/gdml2wrl.cxx.o
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:16:27: error: G4GDMLParser.hh: No such file or directory
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:17:32: error: G4VPhysicalVolume.hh: No such file or directory
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:18:30: error: G4LogicalVolume.hh: No such file or directory
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:19:28: error: G4PVPlacement.hh: No such file or directory
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:20:23: error: G4VSolid.hh: No such file or directory
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:21:27: error: G4Polyhedron.hh: No such file or directory
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:22:24: error: G4Point3D.hh: No such file or directory
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:24: error: variable or field 'visit' declared void
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:24: error: 'G4VPhysicalVolume' was not declared in this scope
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:24: error: 'pv' was not declared in this scope
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:24: error: 'G4LogicalVolume' was not declared in this scope
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:24: error: 'lv' was not declared in this scope
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:24: error: 'G4int' was not declared in this scope
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:24: error: initializer expression list treated as compound expression
    /home/blyth/e/geant4/geometry/gdml/gdml2wrl/gdml2wrl.cxx:25: error: expected ',' or ';' before '{' token
    make[2]: *** [CMakeFiles/gdml2wrl.dir/gdml2wrl.cxx.o] Error 1
    make[1]: *** [CMakeFiles/gdml2wrl.dir/all] Error 2
    make: *** [all] Error 2



