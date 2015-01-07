G4DAEChroma Implementation Details
====================================

.. contents:: :local:


G4DAEChroma Dependencies
--------------------------

Geant4
   Simulation Toolkit 

ZMQ
   ZeroMQ message queue transport 

cJSON
   JSON string serialization/deserialization

numpy.hpp
   NumPy array serialization/deserialization

RapSQLite
   Interface to SQLite3 database, for control/monitoring.

CNPY
   NumPy array serialization/deserialization, 
   (to be eliminated by migration to numpy.hpp)


Former dependency on ROOT for TObject serialization
has been eliminated. 


High Level API
----------------

`G4DAEChroma : G4DAEManager` 
      high level API for collection of Geant4 data and external processing 
      and integration back into propagation

`G4DAEManager`
      task control and timing 


Collectable/Transportable Object Definitions
----------------------------------------------

Template type providing type specifics, describing 
shape and content of arrays and template envvars 
that standardize location of persisted lists. 

::

    G4DAEPhoton.hh
    G4DAECerenkovPhoton.hh
    G4DAECerenkovStep.hh
    G4DAEScintillationStep.hh
    G4DAEScintillationPhoton.hh
    G4DAEPmtHit.hh
    G4DAEProp.hh


Collectable/Transportable Lists
--------------------------------------------

Template specializations of `G4DAEList` eg::

    typedef G4DAEList<G4DAEPhoton> G4DAEPhotonList 
    typedef G4DAEList<G4DAECerenkovStep> G4DAECerenkovStepList 

::

    G4DAEPhotonList.hh
    G4DAECerenkovPhotonList.hh
    G4DAECerenkovStepList.hh
    G4DAEChromaPhotonList.hh
    G4DAEScintillationStepList.hh
    G4DAEScintillationPhotonList.hh
    G4DAEPmtHitList.hh
    G4DAEPropList.hh


Manual Lists to be migrated to common machinery 
---------------------------------------------------

`G4DAEHit`
      serializable single hit struct, 
      initialized from indexed elements of `G4DAEPhotons` 
      with local `G4AffineTransform` applied separately 

`G4DAEHitList : G4DAEArrayHolder`
      adding a `G4DAEHit` immediately serializes into constituent `G4DAEArray`


Transport infrastructure
---------------------------------

`G4DAETransport`
      holder of various objects and socket for sending/receiving them 

`G4DAESocket<T> : G4DAESocketBase` 
      Provides separate Send and Receive methods, using template type constructors

`G4DAESocketBase` 
      `G4DAESerializable* SendReceiveObject(G4DAESerializable* request) const`

`G4DAESerializable`
      interface to allow instance transport via `G4DAESocket`


Array Collection and Transport 
--------------------------------

`G4DAEList<T> : G4DAEArrayHolder` 
      implements `G4DAESerializable` using capabilities of constituent `G4DAEArray`

`G4DAEArrayHolder : G4DAESerializable`
      combination of a `G4DAEArray` and `G4DAEMetadata` constituents allowing 
      arrays with metadata to be transported via `G4DAESocket`

`G4DAEArray : G4DAESerializable` 
      NumPy (NPY) array serialization/deserialization using numpy.hpp. 
      **Effectively allows numpy arrays to be created from C++**
       
`G4DAEMetadata : G4DAESerializable` 
      Between process/language communication using JSON strings,  
      python dicts accessible to C++ as `map<string,string>` 

Config and control Infrastructure
-----------------------------------

`G4DAEDatabase`
      access for inserts and queries to sqlite3 database using **RapSQLite** 

Utililites
------------

`G4DAEBuffer`
      simple byte holder

`G4DAECommon`
      utility functions for md5digest, zmq transport, buffer dumping 

`G4DAEMap`
      `map<string,string>`

`G4DAETime`
      realtime measurement


Geometry
----------

`G4DAEGeometry`
      creates `G4DAETransformCache` instance by  
      traversing Geant4 volume tree collecting 
      positions (`G4AffineTransform`) and identifiers 
      of sensitive detector volumes

`G4DAETransformCache`  
      Map of SD identifiers to `G4AffineTransform` 
      with serialization/deserialization functionality.
      Allows global hit coordinates to be transformed into local 
      PMT coordinates.
          
      TODO: move from cnpy to numpy.hpp 

`G4DAEMaterialMap`
      Via mapping between material names and codes, provides
      lookup array to translate Geant4 material codes
      into Chroma material codes.


Geant4 Hit Collection Integration
-----------------------------------

`G4DAESensDet : G4VSensitiveDetector` 
      Funnels GPU formed hits into standard Geant4 hit collections.

      Normally hits are formed: 
              `bool ProcessHits(G4Step*,G4TouchableHistory*)`

      Instead bulk hit collection is provided:
              `void CollectHits(G4DAEPhotons*,G4DAETransformCache*)`

      Operates via detector specialized constituent `G4DAECollector` subclass
      which steals pointers to preexisting hit collections 
      (eg formed by DetDesc/Gaudi)

`G4DAECollector`  
      does the work of hit handling, directed by `G4DAESensDet`

      `void CollectHits(G4DAEPhotons*, G4DAETransformCache*)`

      * forms `G4DAEHit` for each returned photon
      * applies local transform for the PmtId obtained from G4DAETransformCache
      * subclass implemented detector specific collection 


`DemoG4DAECollector : G4DAECollector`
      example of detector specialized hit collector implementing
 
     `void Collect(const G4DAEHit& hit)`


Daya Bay specializations from DetSimChroma
--------------------------------------------

From NuWa-trunk/dybgaudi/Simulation/DetSimChroma.

`DybG4DAECollector :  G4DAECollector`
      subclass handling detector specific hit collection operations
      steals hit collection pointers and provides a backdoor to 
      populate them 

`DybG4DAEGeometry : G4DAEGeometry`
      specializing subclass providing sensor id 
      for location in the geometry tree         

      `size_t TouchableToIdentifier( const G4TouchableHistory& hist )` 


G4DAEChroma Config and Initialization
----------------------------------------

Config and Initialization of G4DAEChroma example in `DsChromaRunAction_BeginOfRunAction`
creates and configures constituent instances from envvar strings.

`G4DAETransport`
    envvar configures network or inproc config for ZMQ communication

`G4DAEDatabase`
    envvar configures path to sqlite3 database for performance monitoring 

`G4DAETransformCache`
    When run inside NuWa DybG4DAEGeometry used to create G4DAETransformCache, 
    which is persisted to file. When run outside NuWa loads cache from file.
    This facilitates mocknuwa running, for fast development cycle.

`G4DAESensDet`
    Trojan sensdet that targets victim by name (eg "DsPmtSensDet").
    The hit collection pointers of the victim are stolen. 
    SensDet registered with Geant4 to gain access to per event 
    hit collections.

`DybG4DAECollector`
     Provides detector specific hit collection handling, routing  
     hits to the appropriate collections.
     


Soon to be removed
---------------------

::

    G4DAETransportCPL.hh
    G4DAEPhotonListOld.hh
    G4DAEPhotons.hh


