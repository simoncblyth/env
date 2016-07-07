# === func-gen- : optickstute/okt fgp optickstute/okt.bash fgn okt fgh optickstute
okt-src(){      echo optickstute/okt.bash ; }
okt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(okt-src)} ; }
okt-vi(){       vi $(okt-source) ; }
okt-env(){      elocal- ; }
okt-usage(){ cat << EOU

Opticks Tutorial Preparation Notes
===================================

TODO
----

Populate the docs http://simoncblyth.bitbucket.org/opticks/

* proj descriptions from the bash usage functions
  these are currently mostly devnotes, separate those out 
  and replace with short user descriptions 


Inspiration
-------------

* http://www-public.slac.stanford.edu/geant4/PastEvents.asp


OpenMesh Tutorial content 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00066.html


What to get users to do ?
--------------------------

* need code samples doing different things

* need analysis of simulated photon data, using numpy 
  to make some plots



Ideas
------

* installation screen capture video 


Installation Questions
------------------------

* fraction of users with machine, breakdown Linux/macOS/Windows ? What versions ?
* how many CUDA capable GPU machines ?







EOU
}
okt-dir(){ echo $(local-base)/env/optickstute/optickstute-okt ; }
okt-cd(){  cd $(okt-dir); }
okt-mate(){ mate $(okt-dir) ; }
okt-get(){
   local dir=$(dirname $(okt-dir)) &&  mkdir -p $dir && cd $dir

}
