G4DAECHROMA transport testing
================================

Same Node Success
-------------------

Client sends workload to broker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 ~]$ czrt-nsend
    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG] config [tcp://127.0.0.1:5001] 
    ChromaPhotonList::Print UID [0] [6]
    ZMQRoot::SendObject sent bytes: 457 
    ZMQRoot::ReceiveObject received bytes: 457 
    ZMQRoot::ReceiveObject reading TObject from the TMessage 
    ZMQRoot::ReceiveObject returning TObject 
    ReceiveObject
    ChromaPhotonList::Print UID [0] [6]
    ChromaPhotonList::Print UID [0] [6]
    ChromaPhotonList::Details [6]
     index 0 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 1 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 2 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 3 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 4 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
     index 5 pos (1,1,1) mom (2,2,2) pol (3,3,3) _t 0 _wavelength 100 _pmtid 101
    [blyth@belle7 ~]$ 


Worker polls the broker for smth to do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 g4daeview]$ g4daechroma.sh
    ZMQ_BROKER_URL_FRONTEND=tcp://203.64.184.126:5001
    ZMQ_BROKER_URL_BACKEND=tcp://203.64.184.126:5002
    ZMQROOT_LIB=/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libZMQRoot.so
    ZMQROOT_LIBRARIES=ZMQRoot
    ZMQROOT_PREFIX=/data1/env/local/env/zmqroot
    CHROMAPHOTONLIST_PREFIX=/data1/env/local/env/chroma/ChromaPhotonList
    CHROMA_GEANT4_SDIR=/src/geant4.9.5.p01
    CHROMAPHOTONLIST_LIB=/data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/i686-slc5-gcc41-dbg/lib/libChroma.so
    Warning in <TEnvRec::ChangeValue>: duplicate entry <Library.vector<char>=vector.dll> for level 0; ignored
    Warning in <TEnvRec::ChangeValue>: duplicate entry <Library.vector<short>=vector.dll> for level 0; ignored
    Warning in <TEnvRec::ChangeValue>: duplicate entry <Library.vector<unsigned-int>=vector.dll> for level 0; ignored
    main
    2014-09-19 13:56:37,296 env.geant4.geometry.collada.g4daeview.g4daechroma:16  polling: DAEDirectResponder connect tcp://203.64.184.126:5002  
    2014-09-19 13:56:37,297 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 0 
    2014-09-19 13:56:38,397 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 1 
    2014-09-19 13:56:39,498 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 2 
    2014-09-19 13:56:40,599 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 3 
    2014-09-19 13:56:41,700 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 4 
    2014-09-19 13:56:42,801 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 5 
    2014-09-19 13:56:43,902 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 6 
    2014-09-19 13:56:45,003 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 7 
    2014-09-19 13:56:46,104 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 8 
    2014-09-19 13:56:47,205 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 9 
    2014-09-19 13:56:47,205 env.zmqroot.responder:131 recv_object req of length 457 <zmq.backend.cython.message.Frame object at 0x90a265c> 
    2014-09-19 13:56:47,391 env.geant4.geometry.collada.g4daeview.daedirectresponder:38  responder reply <ROOT.ChromaPhotonList object ("ChromaPhotonList") at 0x9eca2f0> 
    <ROOT.ChromaPhotonList object ("ChromaPhotonList") at 0x9eca2f0>
    <class 'env.zmqroot.serialize.ChromaPhotonList'>
    [6L, 6L, 6L, 6L, 6L, 6L, 6L, 6L, 6L, 6L, 6L, 6L]
    {'pmtid': 101, 'pz': 2.0, 'px': 2.0, 'py': 2.0, 'polz': 3.0, 'poly': 3.0, 'polx': 3.0, 't': 0.0, 'x': 1.0, 'y': 1.0, 'wavelength': 100.0, 'z': 1.0}
    {'pmtid': 101, 'pz': 2.0, 'px': 2.0, 'py': 2.0, 'polz': 3.0, 'poly': 3.0, 'polx': 3.0, 't': 0.0, 'x': 1.0, 'y': 1.0, 'wavelength': 100.0, 'z': 1.0}
    {'pmtid': 101, 'pz': 2.0, 'px': 2.0, 'py': 2.0, 'polz': 3.0, 'poly': 3.0, 'polx': 3.0, 't': 0.0, 'x': 1.0, 'y': 1.0, 'wavelength': 100.0, 'z': 1.0}
    {'pmtid': 101, 'pz': 2.0, 'px': 2.0, 'py': 2.0, 'polz': 3.0, 'poly': 3.0, 'polx': 3.0, 't': 0.0, 'x': 1.0, 'y': 1.0, 'wavelength': 100.0, 'z': 1.0}
    {'pmtid': 101, 'pz': 2.0, 'px': 2.0, 'py': 2.0, 'polz': 3.0, 'poly': 3.0, 'polx': 3.0, 't': 0.0, 'x': 1.0, 'y': 1.0, 'wavelength': 100.0, 'z': 1.0}
    {'pmtid': 101, 'pz': 2.0, 'px': 2.0, 'py': 2.0, 'polz': 3.0, 'poly': 3.0, 'polx': 3.0, 't': 0.0, 'x': 1.0, 'y': 1.0, 'wavelength': 100.0, 'z': 1.0}
    2014-09-19 13:56:47,931 env.zmqroot.responder:147 send_object rep of length 457 <env.zmqroot.serialize.c_char_Array_457 object at 0xb7ecc194> 
    2014-09-19 13:56:48,931 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 10 
    2014-09-19 13:56:50,031 env.geant4.geometry.collada.g4daeview.g4daechroma:19  polling 11 



Normal Topology Success, via tunneling both client and worker
-----------------------------------------------------------------

* broker on N, `czmq_broker` running all the time under supervisord
* worker on D, interactively started after which polls every second::

   delta:~ blyth$ g4daechroma.sh --zmqtunnelnode=N 

* client on D or N, interactively sends CPL at each invokation::

   delta:~ blyth$ czrt.sh --zmqtunnelnode=N 
   [blyth@belle7 ~]$ czrt-nsend 







