Tasks
======

.. contents:: :local:


Geant4 Profiling [DONE]
------------------------

* verify expectations of where CPU time is spent, check what the potential gains really are
* :doc:`/muon_simulation/profiling/base/index`

Convert Detector Geometry from Solid to Surface representation
---------------------------------------------------------------

Requirement
~~~~~~~~~~~~

Convert Geant4 geometry representation of solids and materials 
into surfaces (triangles) with inside/outside materials for each triangle. 
This simpler representation is needed in order to bring geometry onto GPU.

Geant4 Exports [DONE]
~~~~~~~~~~~~~~~~~~~~~~

Learn details of existing Geant4 geometry functionality, to see what 
can be reused.

* geant4 export VRML2 with VRML2FILE driver into a .wrl file 

  * :doc:`/geant4/geometry/vrml2/index` 
  * precision issues in export observed
  
    * TODO: check if already fixed in later geant4, otherwise submit patch 

* geant4 export GDML 

  * DONE :doc:`/geant4/geometry/gdml/index` 
  * truncation of volume names observed+fixed

    * TODO: check if already fixed in later geant4, otherwise submit patch 

Geant4 Collada exporter [DONE]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create Geant4 Collada(widely supported 3D format) exporter. 
Using a form of geometry representation that can easily be converted 
to the STL(a very simple 3D format) needed by Chroma. 

* Actually probably pycollada can be used to allow Chroma to directly access triangles from .dae files

Collada is in some sense intermediate to VRML2 and GDML, 
so the new exporter can draw upon those existing exporters.

G4DAE Improvements
~~~~~~~~~~~~~~~~~~~~

#. avoid pointers in volume id 
#. workout how to visually distinguish surface types in COLLADA

Meshlab COLLADA loading [ABANDONED]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. added bbox caching in order to allow node navigation with external control via DBUS 

   * rearrange to allow absolute id addressing, otherwise cannot jump to a volume by 
     absolute index unless operating from the full geometry (which is too slow)
   * you might think that relative indexing from the index of the subroot would 
     allow this, but that only works if full depth is used 

Web Visualisation Improvements [ABANDONED]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. need development machine that supports WebGL to make progress on this

#. Abandoned, as pyopengl/glumpy provides a nicer environment in which to proceed
   and especially as its then easy to integrate with Chroma/pycuda

Alternatives [ABANDONED] 
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternate workflows for G4 export and mesh conversion

* HepRep ? 
* other geometry libraries: CGAL, BRL-CAD, ... 

Decided that best to start from Geant4 and develop in that 
context initially.

Meshlab Visualisation [ABANDONED]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. added external numerical control via DBUS 
#. operating inside messy student developed code, can only go so far



Blender Visualisation [NOPE]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blender GUI is unusable.


MeshLab conversions ? [DECIDED AGAINST]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* meshlab imports COLLADA and VRML2 :doc:`/graphics/meshlab/index`
* meshlab export STL 
* mesh visualization with meshlab, blender, freewrl 

Although meshlab can convert VRML2 into STL, this misses 
the material information. 

Meshlab turns out to be extremely slow at loading COLLADA (40 min for full geometry). 
Even with my fixes as available from bitbucket it is still too slow (10 min).
Also MeshLab (and underlying VCGLIB) dependency on Qt makes it difficult to 
use widely (especially older linux).

Thus best to use MeshLab for visualization only.

* https://bitbucket.org/scb-/meshlab/overview

Geometry Validation [ON GOING]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Chroma surface property handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* surface properties, retaining volume/surface identity into a mesh representation 


Chroma Installation
--------------------

Need an easily reproducible installation approach for NuWa+Chroma 
and its many dependencies http://chroma.bitbucket.org/install/details.html
Extending `dybinst` and `NuWa/LCG_Builders` presumably the way to go,  

  * probably the CUDA toolkit needs to be excluded, using the local installs

Chroma is aggressive about versions, how critical these are is unknown

  * `GEANT4.9.5 or later` [lots of work needed to bring all of geant4 up to 4.9.5, maybe just patches for issues?]
  * `ROOT 5.32 or later` 

Geant4/Chroma integration
---------------------------

* :doc:`/chroma/chroma_geant4_integration`

grab cohort of optical photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
StackAction better than TrackingAction (currently used), advantages:

   * "interestingness" optimisation, only propagate OP for interesting events
   * delay OP tracks, collecting their parameters then give them back modified to be just before step onto sensitive detector volumes 
 
parallel GPU transport 
~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`/chroma/chroma_physics`
* parallel propagate the cohort of OP

give back to G4 at sensitive detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
Need seemless integration with the rest of the reconstruction chain

maybe more general approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drop in replacement for some Geant4 classes which provide 
the GPU acceleration with minimal disturbance.  
Perhaps:

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
* NuWa settings http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/1397#comment:7


Glossary
---------

OP
    Geant4 Optical Photons are distinct from Gammas, assigned special PDG code 20022


