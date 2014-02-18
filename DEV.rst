Development Log
===========================

Feb 18, 2014
--------------

Expose G4MaterialPropertiesTable maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. :doc:`geant4/geant4_patch` 

Rebuild NuWa Geant4
~~~~~~~~~~~~~~~~~~~~~~

Rebuild Geant4 on N with the API exposed/

Test export run
~~~~~~~~~~~~~~~~~


Use API
~~~~~~~~~

Compare new geant4 GDML with the version used with NuWa 

* /usr/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/include

::

    src/G4DAEWriteMaterials.cc:144: error: cannot convert 
    'const std::map<G4String, G4MaterialPropertyVector*, std::less<G4String>, std::allocator<std::pair<const G4String, G4MaterialPropertyVector*> > >*' to 
    'const std::map<G4String, G4PhysicsOrderedFreeVector*, std::less<G4String>, std::allocator<std::pair<const G4String, G4PhysicsOrderedFreeVector*> > >*' in initialization

Future geant4 gets rid of the deficient G4MaterialPropertyVector typedefing from G4PhysicsOrderedFreeVector.
/usr/local/env/geant4/geant4.10.00.b01/source/materials/include/G4MaterialPropertyVector.hh::

     56 #include "G4PhysicsOrderedFreeVector.hh"
     ..
     62 typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;





