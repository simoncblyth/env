DetSimChroma
===============

DsChromaRunAction
------------------

At BeginOfRunAction G4DAEChroma and constituents 
are configured/initialized based on python settable properties. 
Several properties name envvars which are read to
get config values. 


DsChromaRunAction properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python settable, but actually little need to change these. 
More likely to need to change the envvars that they name.


transport
     Name of envvar holding the transport config string, default G4DAECHROMA_CLIENT_CONFIG

sensdet
     Name of target SD where Chroma derived hits will be collected, default DsPmtSensDet

cachekey
     Name of envvar pointing to directory where cache is written to, default G4DAECHROMA_CACHE_DIR

databasekey
     Name of envvar pointing to config/monitoring sqlite3 database path, default G4DAECHROMA_DATABASE_PATH

TouchableToDetelem
     The ITouchableToDetectorElement to use to resolve sensor, default TH2DE

PackedIdPropertyName
     The name of the user property holding the PMT ID, default PmtID


DsChromaRunAction::BeginOfRunAction
-------------------------------------

G4DAEChroma singleton instanciated and constituents ctored

* G4DAETransport, manages sending/receiving photons/hits via ZeroMQ networking 

* G4DAEDatabase, config and monitoring sqlite3 database

* G4DAETransformCache, created by DybG4DAEGeometry by traversing Geant4 geometry tree
  allows global GPU hit coordinates to be transformed to local coordinates appropriate 
  to the sensors they landed on 

* G4DAESensDet, steals pointers to standard DetDesc hit collections 
  allowing GPU hits to be added in bulk

* DybG4DAECollector, converts GPU hits into standard SimHits  



DsChromaStackAction
---------------------


NeutronParent

PhotonKill

MaxPhoton

ModuloPhoton

ChromaPropagate
      








