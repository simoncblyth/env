History
=========

* https://wiki.bnl.gov/dayabay/index.php?title=Weighted_optical_photons

DavidJ
-------

Nicely documented study, a good starting point. 

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSim/src/DsFastMuonSimulation.cc
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSim/src/DsFastMuonStackAction.h
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSim/src/DsFastMuonStackAction.cc

* http://dayabay.ihep.ac.cn/tracs/dybsvn/ticket/953
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/people/djaffe/OPW/fmcpmuon.py

Zhe
-----

* http://dayabay.ihep.ac.cn/DocDB/0038/003894/001/FastMuon_Scale.pdf

* Scale down the number of optical photons generated in AD 

::

      N_OP > Threshold

      For every N OP: Kill N-1 OPs 
      Increase the weight of remaining OP by N

Let 1 OP of higher weight stand in for N of them

The rescale is done in Geant4 StackingAction 
which means OPsare already created and only  
the propagation is suppressed. Further work can 
move the compression into scintillation process to 
suppress the creation of them. 


Maxim/Kevin
------------

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSim/src/DsOpStackAction.cc




