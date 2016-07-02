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

* Update env notes.  http://simoncblyth.bitbucket.org/env/notes/

* bitbucketstatic- 


* There is too much there to be useful docs. 
* Look into opticks migration : just for docs ?


Inspiration
-------------

* http://www-public.slac.stanford.edu/geant4/PastEvents.asp

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
