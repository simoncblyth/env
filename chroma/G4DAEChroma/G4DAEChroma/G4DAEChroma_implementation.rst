G4DAEChroma Implementation Details
====================================

Umbrella
-----------

`G4DAEChroma` 
      high level API for photon collection and propagation

Photon transport infrastructure
---------------------------------


`G4DAETransport`
      holder of photons, hits and socket for sending/receiving them 

`G4DAESocket<T> : G4DAESocketBase` 
      Provides separate Send and Receive methods, using template type constructors

`G4DAESocketBase` 
      `G4DAESerializable* SendReceiveObject(G4DAESerializable* request) const`

`G4DAEArray : G4DAESerializable` 
      basis for NPY serialization/deserialization using numpy.hpp

`G4DAEArrayHolder` 
      common functionality for holding constituent `G4DAEArray` 

`G4DAEBuffer`
      simple byte holder

`G4DAECommon`
      utility functions for md5digest, zmq transport, buffer dumping 


Photon lists
--------------

`G4DAEPhotonList : G4DAEPhotons` 
      implements `G4DAESerializable` using capabilities of constituent `G4DAEArray`

`G4DAEChromaPhotonList : G4DAEPhotons`
      old ROOT dependant implementation based on constituent `ChromaPhotonList` 
      implements `G4DAESerializable` using ROOT `TObject` serialization and `G4DAEBuffer`  

`G4DAEPhotons : G4DAESerializable`
      interface for photon list classes

`G4DAESerializable`
      interface to allow instance transport via `G4DAESocket`


Config and control Infrastructure
-----------------------------------

`G4DAEDatabase`
      access for inserts and queries to sqlite3 database using **RapSQLite** 

`G4DAEMetadata : G4DAESerializable` 
      Between process/language communication using JSON strings,  
      python dicts accessible to C++ as `map<string,string>` 

Geometry
----------

`G4DAEGeometry`
      creates `G4DAETransformCache` instance by  
      traversing Geant4 volume tree collecting 
      positions (`G4AffineTransform`) and identifiers 
      of sensitive detector volumes

`G4DAETransformCache`  
      Map of SD identifiers to `G4AffineTransform` 
      with serialization/deserialization functionality
          
      TODO: move from cnpy to numpy.hpp 

Hits
-----

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

`G4DAEHitList : G4DAEArrayHolder`
      adding a `G4DAEHit` immediately serializes into constituent `G4DAEArray`

`G4DAEHit`
      serializable single hit struct, 
      initialized from indexed elements of `G4DAEPhotons` 
      with local `G4AffineTransform` applied separately 


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
     



