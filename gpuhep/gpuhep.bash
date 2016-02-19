# === func-gen- : gpuhep/gpuhep fgp gpuhep/gpuhep.bash fgn gpuhep fgh gpuhep
gpuhep-src(){      echo gpuhep/gpuhep.bash ; }
gpuhep-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gpuhep-src)} ; }
gpuhep-vi(){       vi $(gpuhep-source) ; }
gpuhep-env(){      elocal- ; }
gpuhep-usage(){ cat << EOU

GPU Usage In HEP
===================

* :google:`GPU Computing in High Energy Physics`

GPUHEP 2014
------------

* https://agenda.infn.it/conferenceDisplay.py?confId=7534
* http://www-library.desy.de/preparch/desy/proc/proc14-05/40.pdf


* http://inspirehep.net/record/1387408

  Proceedings, GPU Computing in High-Energy Physics (GPUHEP2014) : 
  Pisa, Italy, September 10-12, 2014
  Claudio Bonati (ed.) , Gianluca Lamanna (ed.) , Massimo D'Elia (ed.) , Marco Sozzi (ed.) 
  2015 - 255 pages

  * http://www-library.desy.de/preparch/desy/proc/proc14-05.pdf



* https://agenda.infn.it/getFile.py/access?contribId=5&sessionId=11&resId=0&materialId=slides&confId=7534

  Sampling Secondary Particles in High Energy Physics Simulation on the GPU
  Soon Yung Jun, Fermilab
  for the Geant Vector/Coprocessor R&D Team



EOU
}
gpuhep-dir(){ echo $(local-base)/env/gpuhep/gpuhep-gpuhep ; }
gpuhep-cd(){  cd $(gpuhep-dir); }
gpuhep-mate(){ mate $(gpuhep-dir) ; }
gpuhep-get(){
   local dir=$(dirname $(gpuhep-dir)) &&  mkdir -p $dir && cd $dir

}
