GDML from Geant4Py
====================

Attempt to access the Polyhedron from Geant4Py failed.
Unsure if just an omission. Need to learn more `boost_python` to try adding it to G4Py
but meanwhile just do it in C++.

This means need to use collada dom for dae creation rather than pycollada.
 

::

    In [10]: vso = lv.GetSolid()

    In [11]: vso
    Out[11]: <Geant4.G4geometry.G4Box object at 0x8acf8b4>

    In [12]: p vso
    -----------------------------------------------------------
        *** Dump for solid - WorldBox ***
        ===================================================
     Solid type: G4Box
     Parameters: 
        half length X: 2.4e+06 mm 
        half length Y: 2.4e+06 mm 
        half length Z: 2.4e+06 mm 
    -----------------------------------------------------------

::

    In [21]: Geant4.G4VSolid?
    Type:           class
    Base Class:     <type 'Boost.Python.class'>
    String Form:    <class 'Geant4.G4geometry.G4VSolid'>
    Namespace:      Interactive
    File:           /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/environments/g4py/lib/Geant4/G4geometry.so
    Docstring:
        solid class

    Constructor information:
    Docstring:
        Raises an exception
        This class cannot be instantiated from Python




::


    [blyth@cms01 source]$ find . -name 'G4VSolid.hh'
    ./geometry/management/include/G4VSolid.hh
    [blyth@cms01 source]$ vi geometry/management/include/G4VSolid.hh
    [blyth@cms01 source]$ find . -name 'G4Box.hh'
    ./geometry/solids/CSG/include/G4Box.hh
    [blyth@cms01 source]$ vi geometry/solids/CSG/include/G4Box.hh
    [blyth@cms01 source]$ find . -name 'G4CSGSolid.hh'
    ./geometry/solids/CSG/include/G4CSGSolid.hh
    [blyth@cms01 source]$ vi geometry/solids/CSG/include/G4CSGSolid.hh


$DYB/external/build/LCG/geant4.9.2.p01/environments/g4py/source/geometry/pyG4VSolid.cc::

     33 #include <boost/python.hpp>
     34 #include "G4Version.hh"
     35 #include "G4VSolid.hh"
     36 
     37 using namespace boost::python;
     38 
     39 // ====================================================================
     40 // module definition
     41 // ====================================================================
     42 void export_G4VSolid()
     43 {
     44   class_<G4VSolid, G4VSolid*, boost::noncopyable>
     45     ("G4VSolid", "solid class", no_init)
     46     // ---
     47     .def("GetName",        &G4VSolid::GetName)
     48     .def("SetName",        &G4VSolid::SetName)
     49     .def("DumpInfo",       &G4VSolid::DumpInfo)
     50 
     51     .def("GetCubicVolume",    &G4VSolid::GetCubicVolume)
     52 #if G4VERSION_NUMBER >=820
     53     .def("GetSurfaceArea",    &G4VSolid::GetSurfaceArea)
     54 #endif
     55 #if G4VERSION_NUMBER >=800
     56     .def("GetPointOnSurface", &G4VSolid::GetPointOnSurface)
     57 #endif
     58     // operators
     59     .def(self == self)
     60     ;
     61 }





::

    In [50]: Geant4.G4geometry.
    Geant4.G4geometry.CreateBox                Geant4.G4geometry.CreateTrd                Geant4.G4geometry.G4FieldManager           Geant4.G4geometry.G4Sphere                 Geant4.G4geometry.G4UnionSolid             Geant4.G4geometry.__name__
    Geant4.G4geometry.CreateCons               Geant4.G4geometry.CreateTubs               Geant4.G4geometry.G4GeometryManager        Geant4.G4geometry.G4SubtractionSolid       Geant4.G4geometry.G4VPhysicalVolume        Geant4.G4geometry.__new__
    Geant4.G4geometry.CreateEllipsoid          Geant4.G4geometry.CreateTwistedBox         Geant4.G4geometry.G4Hype                   Geant4.G4geometry.G4Tet                    Geant4.G4geometry.G4VSolid                 Geant4.G4geometry.__package__
    Geant4.G4geometry.CreateEllipticalCone     Geant4.G4geometry.CreateTwistedTap         Geant4.G4geometry.G4IntersectionSolid      Geant4.G4geometry.G4Torus                  Geant4.G4geometry.G4VTouchable             Geant4.G4geometry.__reduce__
    Geant4.G4geometry.CreateEllipticalTube     Geant4.G4geometry.CreateTwistedTrd         Geant4.G4geometry.G4LogicalVolume          Geant4.G4geometry.G4TouchableHistory       Geant4.G4geometry.__G4MagneticField        Geant4.G4geometry.__reduce_ex__
    Geant4.G4geometry.CreateHype               Geant4.G4geometry.CreateTwistedTubs        Geant4.G4geometry.G4MagneticField          Geant4.G4geometry.G4TransportationManager  Geant4.G4geometry.__class__                Geant4.G4geometry.__repr__
    Geant4.G4geometry.CreateOrb                Geant4.G4geometry.G4BooleanSolid           Geant4.G4geometry.G4Navigator              Geant4.G4geometry.G4Trap                   Geant4.G4geometry.__delattr__              Geant4.G4geometry.__setattr__
    Geant4.G4geometry.CreatePara               Geant4.G4geometry.G4Box                    Geant4.G4geometry.G4Orb                    Geant4.G4geometry.G4Trd                    Geant4.G4geometry.__dict__                 Geant4.G4geometry.__sizeof__
    Geant4.G4geometry.CreatePolycone           Geant4.G4geometry.G4ChordFinder            Geant4.G4geometry.G4PVPlacement            Geant4.G4geometry.G4Tubs                   Geant4.G4geometry.__doc__                  Geant4.G4geometry.__str__
    Geant4.G4geometry.CreatePolyhedra          Geant4.G4geometry.G4Cons                   Geant4.G4geometry.G4PVReplica              Geant4.G4geometry.G4TwistedBox             Geant4.G4geometry.__file__                 Geant4.G4geometry.__subclasshook__
    Geant4.G4geometry.CreateSphere             Geant4.G4geometry.G4Ellipsoid              Geant4.G4geometry.G4Para                   Geant4.G4geometry.G4TwistedTrap            Geant4.G4geometry.__format__               
    Geant4.G4geometry.CreateTet                Geant4.G4geometry.G4EllipticalCone         Geant4.G4geometry.G4Polycone               Geant4.G4geometry.G4TwistedTrd             Geant4.G4geometry.__getattribute__         
    Geant4.G4geometry.CreateTorus              Geant4.G4geometry.G4EllipticalTube         Geant4.G4geometry.G4Polyhedra              Geant4.G4geometry.G4TwistedTubs            Geant4.G4geometry.__hash__                 
    Geant4.G4geometry.CreateTrap               Geant4.G4geometry.G4Field                  Geant4.G4geometry.G4Region                 Geant4.G4geometry.G4UniformMagField        Geant4.G4geometry.__init__        


