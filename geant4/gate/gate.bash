# === func-gen- : geant4/gate/gate fgp geant4/gate/gate.bash fgn gate fgh geant4/gate
gate-src(){      echo geant4/gate/gate.bash ; }
gate-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gate-src)} ; }
gate-vi(){       vi $(gate-source) ; }
gate-env(){      elocal- ; }
gate-usage(){ cat << EOU

GATE 
======

Simulations of Preclinical and Clinical Scans in Emission Tomography,
Transmission Tomography and Radiation Therapy

* http://www.opengatecollaboration.org


* http://wiki.opengatecollaboration.org/index.php/Users_Guide_V8.0


GATE Paper
-----------------

* http://www.opengatecollaboration.org/sites/default/files/Jan2004.pdf
* ~/opticks_refs/OpenGate_G4_Jan2004.pdf

GATE: a simulation toolkit for PET and SPECT

* PET: Positron emission tomography
* SPECT : Single-photon emission computed tomography





GATE : Optical Photon
------------------------

* http://wiki.opengatecollaboration.org/index.php/Users_Guide_V8.0:Generating_and_tracking_optical_photons

Before discussing how to use the optical photon tracking, it has to be
mentioned that there are a few disadvantages in using optical transport. First,
the simulation time will increase dramatically. For example, most scintillators
used in PET generate in the order of 10,000 optical photons at 511 keV, which
means that approximately 10,000 more particles have to be tracked for each
annihilation photon that is detected. Although the tracking of optical photons
is relatively fast, a simulation with optical photon tracking can easily be a
factor thousand slower than one without. Finally, in order to perform optical
simulations, many parameters are needed for the materials and surfaces, some of
which may be difficult to determine.








EOU
}
gate-dir(){ echo $(local-base)/env/geant4/gate/geant4/gate-gate ; }
gate-cd(){  cd $(gate-dir); }
gate-mate(){ mate $(gate-dir) ; }
gate-get(){
   local dir=$(dirname $(gate-dir)) &&  mkdir -p $dir && cd $dir

}
