
Umbrella::

     G4DAEChroma 

     G4DAETransport

Geometry::

     G4DAEGeometry

     G4DAETransformCache 

Hits::

     DemoG4DAECollector : G4DAECollector  

     G4DAECollector  

     G4DAEHit

     G4DAESensDet : G4VSensitiveDetector 

     G4DAEHitList : G4DAEArrayHolder 

Interface definitions::

     G4DAEPhotons : G4DAESerializable 
          interface for photon list classes

     G4DAESerializable
          interface to allow instance transport via G4DAESocket

Photon lists::

     G4DAEPhotonList : G4DAEPhotons 
           new NPY implementation
           implements G4DAESerializable using capabilities of constituent G4DAEArray

     G4DAEChromaPhotonList : G4DAEPhotons
           old ROOT dependant implementation based on constituent ChromaPhotonList 
           implements G4DAESerializable using ROOT TObject serialization and G4DAEBuffer  

Infrastructure::

     G4DAEMetadata : G4DAESerializable 

     G4DAEArray : G4DAESerializable 
           NPY numpy serialization 

     G4DAESocketBase 
           G4DAESerializable* SendReceiveObject(G4DAESerializable* request) const

     G4DAESocket<T> : G4DAESocketBase 
           Provides separate Send and Receive methods, using template type constructors
           TODO: check if this can be removed

     G4DAEArrayHolder 

     G4DAEBuffer
           simple byte holder

     G4DAECommon
           utility functions for md5digest, zmq transport, buffer dumping 


TO REMOVE::

     G4DAETransportCPL

     Photons_t






