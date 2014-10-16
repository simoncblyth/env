
G4DAEChroma
=============

Objective
------------

Pull out everything Chroma related and reusable 
from StackAction and SensitiveDetector
for flexible reusability in different Geant4 contexts

Dependencies
------------

* giga/gaudi/gauss NOT ALLOWED 
* sticking to plain Geant4, ZMQ, ZMQRoot,... for generality 

Issues
--------

Development Cycle too slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: Create test application for machinery test 
(enable to rapidly work on the marshalling) 

* reads Dyb geometry into G4 from exported GDML
* reads some initial photon positions from a .root file
* invokes this photon collection and propagation 
* dumps the hits returned


GPU Hit handling
~~~~~~~~~~~~~~~~~

* how to register DsChromaPmtSensDet instead of (or in addition to) DsPmtSensDet
  or some how get access to DsPmtSensDet

  * class name "DsPmtSensDet" is mentioned in DetDesc 
    logvol sensdet attribute, somehow DetDesc/GiGa 
    hands that over to Geant4 : need to swizzle OR add ? 

  * old approach duplicated bits of "DsPmtSensDet" for adding 
    hits into the StackAction : that was too messy then, but perhaps
    clean enough now have pulled out Chroma parts into G4DAEChroma 

  * but needs access to private methods from DsPmtSensDet, so 
    maybe a no-no anyhow : especially as need very little
    functionality 

* how to get access to DsPmtSensDet in order to add hits

  * provide singleton accessor for cheat access to globally 
    shared instance ? 
    Approach has MT complications : but no need to worry about that yet

  * gaudi has a way of accessing the instance, do it externally (where?)
    and pass it in 

* how to handle hits interfacing to detector specific code



