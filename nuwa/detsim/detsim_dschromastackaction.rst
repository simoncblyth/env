DetSim ChromaStackAction
=========================

Dependencies
--------------

ZMQRoot
~~~~~~~~~

ZMQRoot is general purpose glue code giving ROOT TObjects ZeroMQ transport capabilities.
Its something that should eventually be part of ROOT.

::

    (chroma_env)delta:zmqroot blyth$ find . 
    .
    ./__init__.py
    ./CMakeLists.txt
    ./include
    ./include/MyTMessage.hh
    ./include/MyTMessage_LinkDef.h
    ./include/ZMQRoot.hh
    ./responder.py
    ./serialize.py
    ./src
    ./src/MyTMessage.cc
    ./src/ZMQRoot.cc
    ./zmqroot.bash

Depends on:

* ROOT TObject/TMessage, needs dictionary generation
* ZMQ C library 


How to make usable from NuWa detsim ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Model after `NuWa-trunk/dybgaudi/Utilities/HistMan` (simple ROOT dependant pkg with dict generation)
#. `-DWITH_ZMQ` preprocessor protect use of `zmq.h` with do nothing stubs, for transitionally not breaking build
#. add ZMQ lcgcmt external
#. can skip the python, thats not needed at client end : the producer of TObjects being from C++

::

    [blyth@belle7 dybx]$ find NuWa-trunk/dybgaudi -name 'requirements' -exec grep -H HistMan {} \;
    NuWa-trunk/dybgaudi/Utilities/HistMan/cmt/requirements:package HistMan
    NuWa-trunk/dybgaudi/Utilities/HistMan/cmt/requirements:library       HistMan     *.cc
    NuWa-trunk/dybgaudi/Utilities/HistMan/cmt/requirements:apply_pattern linker_library       library=HistMan
    NuWa-trunk/dybgaudi/Utilities/HistMan/cmt/requirements:apply_pattern install_more_includes more=HistMan
    NuWa-trunk/dybgaudi/Utilities/HistMan/cmt/requirements:apply_pattern reflex_dictionary dictionary=HistMan \
    NuWa-trunk/dybgaudi/Utilities/UtilitiesRelease/cmt/requirements:use HistMan     v*  Utilities
    NuWa-trunk/dybgaudi/DybSvc/StatisticsSvc/cmt/requirements:use HistMan         v*  Utilities


ChromaPhotonList
~~~~~~~~~~~~~~~~~

ChromaPhotonList, simple TObject std::vector photon info collector 

* ROOT 
* Geant4  (could easily be removed)


Usable from Detsim ?
^^^^^^^^^^^^^^^^^^^^^^^

#. Header only class included into `DsChromaStackAction.cc`

   * Nope it needs dictionary generation 

#. expedient: make it travel with Utilities/ZMQRoot ?  As "an example" (this is the quick way).





