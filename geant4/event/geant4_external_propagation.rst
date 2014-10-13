Geant4 External Propagation
=============================

Remit 
------

This page gives **JUST** an overview/summary.
For details, code listings etc.. see: 

* :doc:`/nuwa/detsim/detsim_giga_geant4_handoff`
* :doc:`/nuwa/detsim/detsim_dspmtsensdet`

External Propagation Requirements
------------------------------------

#. collect optical photon data for all generated optical photons
#. suspend other processing of optical photons beyond collection
#. once all (or some configured number) of optical photons are collected 
   dispatch the external propagation and get the results back
#. merge the propagated photon (or maybe hit data) back into geant4 
#. proceed with geant4 processing as if the propagation was standard geant4  


Alternate Early Photon Collection
----------------------------------------

Collection of photon data within the processes  
before becomes a G4Track and being stacked 
would avoid track stacking processing 
and associated memory expense/concerns.  

* to support this split off photon collection into `G4DAEChromaManager`


GPU Hit formation Approach
----------------------------

* `StackAction::ClassifyNewTrack(G4Track*)` 

   * collect photon data (TrackID?) and kill G4Track  

* `StackAction::NewStage` 

   * batched external GPU propagation
   * from ZMQ response form the PMT hits and fill hit collection 
     (requires detector and PMT id info to be gleaned GPU side)
   * try to keep detector specifics CPU side, 
     keep GPU code as generic as possible 
 
Pros
~~~~~

Avoids complications:

* G4Track memory constraints (diddling with placeholder duplicate objects to avoid)  
* how to merge back

Cons
~~~~~~

* detector specific code duplication needed from `DsPmtSensDet::ProcessHit`

How to implement ?
~~~~~~~~~~~~~~~~~~~~~~~

Main issue is forming PMTID

* look into general SDManager traverse to harvest PMTIDs and incorporate into 
  the exported COLLADA geometry

  * remember emergent nature of the geometry traverse, 
    cannot apply identifier labels to all PMTs within the recursive structure 
    but easy to do in a simple separate flat array 

* there is a Chroma slot for a solid identifier ?



Propagated Photon Mergeback approach
---------------------------------------

The Geant4 "allowed" way to move photons gets complicated 
with creation of a pseudo process and creating particle changes.  
That seems expensive and pointless.

Initially preferred this as could allow to avoid a reimplementation
if `ProcessHits`.

pseudo process complications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Processes operate at track level, need to propagate at event (or track collection level) 

* obvious optimization is to throw photons that dont get to PMTs, how to handle


