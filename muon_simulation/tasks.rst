Tasks
======

.. contents:: :local:

Geant4 Profiling
-----------------

* verify expectations of where CPU time is spent, check what the potential gains really are
* DONE :doc:`/muon_simulation/profiling/base/index`


Convert Detector Geometry from Solid to Surface representation
---------------------------------------------------------------

* convert Geant4 geometry into STL mesh needed for Chroma/GPUs

   * geant4 export VRML2 with VRML2FILE driver into a .wrl file 

       * DONE with caveats :doc:`/graphics/geant4/vrml` 

   * meshlab import VRML2 :doc:`/graphics/mesh/meshlab`
   * meshlab export STL 
   * mesh visualization with meshlab, blender, freewrl 


* geometry validation ?

   * visualisation with meshlab, blender
   * surface properties, retaining volume/surface identity into a mesh representation 
   * alternate workflows for G4 export and mesh conversion

       * HepRep ? 
       * other geometry libraries: CGAL, BRL-CAD, ... 


Geant4/Chroma integration
---------------------------

* :doc:`/muon_simulation/chroma/chroma_geant4_integration`

grab cohort of optical photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
StackAction better than TrackingAction (currently used), advantages:

   * "interestingness" optimisation, only propagate OP for interesting events
   * delay OP tracks, collecting their parameters then give them back modified to be just before step onto sensitive detector volumes 
 
parallel GPU transport 
~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`/muon_simulation/chroma/chroma_physics`
* parallel propagate the cohort of OP

give back to G4 at sensitive detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
Need seemless integration with the rest of the reconstruction chain


maybe more general approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drop in replacement for some Geant4 classes which provide 
the GPU acceleration with minimal disturbance.  Perhaps:

   * processes/transportation/src/G4Transportation.cc
   * geometry/navigation/src/G4TransportationManager.cc

Usual Geant4 API approach of eg providing UserStackingAction
requires custom handling. Complications: geometry conversion.

CUDA/Chroma testing
-----------------------------------

* test hardware
* perform standalone Chroma operation tests

Chroma vs G4 Optical Process Validation
----------------------------------------

* establish statistical equivalence between Chroma and G4



Glossary
---------

OP
    Geant4 Optical Photons are distinct from Gammas, assigned special PDG code 20022


