Chroma/NuWa/Geant4 Integration with ZeroMQ
============================================

Related
-------

* :doc:`/nuwa/detsim/detsim_dsopstackaction`
* :doc:`/chroma/chroma_geant4_integration`
* :doc:`/chroma/chroma_nuwa_geant4_integration`
* :doc:`/geant4/event/geant4_stackingaction`
* :doc:`/geant4/event/geant4_stackmanager`


Thoughts
---------

Photon Collection 
~~~~~~~~~~~~~~~~~~~

#. `DsChromaOpStackAction`  (based on  `DsOpStackAction` to create and populate `ChromaPhotonList` structure)

#. `DsOpStackAction` doesnt actually store photons, it just provides classification `fKill/fWaiting/fUrgent`
   that directs G4 which stack to place things on

   * are the G4 stacks accessible without patching ? 
   * convert that into `ChromaPhotonList`

Send to server 
~~~~~~~~~~~~~~~~

#. where to put the ZMQ context ? config via envvar 

Handle Results from server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. 1st try dealing pulling back the propagated `ChromaPhotonList`


NuWa/Geant4 Integration
-------------------------

* `/data1/env/local/dybx/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsOpStackAction.cc`


* :doc:`/geant4/event/geant4_stackingaction`

NuWa DetSim
~~~~~~~~~~~~~

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSim/src



Where is DsOpStackAction hooked up in NuWa ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Simulation/DetSim/genConf/DetSim/DetSimConf.py
    Simulation/DetSim/src/DetSim_entries.cc
    Simulation/DetSim/python/poolgun.py             giga.StackingAction="DsOpStackAction"
    Simulation/DetSim/python/DetSim/OpStack.py      opstack = DsOpStackAction("GiGa.DsOpStackAction") 


GiGa hookup, `Simulation/DetSim/python/DetSim/OpStack.py`::

     80     from DetSim.DetSimConf import DsPhysConsOptical
     81     optical = DsPhysConsOptical("GiGa.GiGaPhysListModular.DsPhysConsOptical")
     82     optical.CerenPhotonScaleWeight = 3.0
     83     optical.ScintPhotonScaleWeight = 3.0
     84     optical.UseScintillation = False
     85 
     86 
     87     from DetSim.DetSimConf import DsOpStackAction
     88     opstack = DsOpStackAction("GiGa.DsOpStackAction")
     89     detsim.giga.StackingAction=opstack
     90     opstack.TightCut = False
     91     opstack.PhotonCut = False
     92     opstack.MaxPhoton = 1e6







How to make ZMQ call from inside C++ code ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





