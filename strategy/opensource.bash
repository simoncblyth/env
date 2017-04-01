# === func-gen- : strategy/opensource fgp strategy/opensource.bash fgn opensource fgh strategy
opensource-src(){      echo strategy/opensource.bash ; }
opensource-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opensource-src)} ; }
opensource-vi(){       vi $(opensource-source) ; }
opensource-env(){      elocal- ; }
opensource-usage(){ cat << EOU

Open Source Usage 
===================


DualContouringSample lessons learned (dcs-)
----------------------------------------------

* https://github.com/nickgildea/DualContouringSample

Integration of DualContouringSample within Opticks 
raised some strategy issues... it was incorporated straight 
into npy- into a subdirectory, and code was developed
both there and within npy-.

Unfortunately far more development than expected was required
to make it usable, which in retrospect means that it was 
a mistake to do that within npy-, especially as 
there are some LGPL concerns.

Also, having the code within the same repo encouraged tight 
coupling : when you really want to promote the opposite
with external code that has LGPL concerns.


SOP : Standard Operating Procedures for Open Source usage
------------------------------------------------------------

* Only very small amounts of open source code (such as PyMCubes) 
  with appropriate license are admissable for direct source inclusion.

* For code from a zip or tarball, create a repo on bitbucket/github
  starting from the asis distribution.
  It can be a private repo on bitbucket if desired.

* Add evaluation code to the repo including a simple interface 
  class that hides the details and just provides what is needed by 
  opticks without using Opticks headers (eg use glm::vec3 for vertices, 
  triangles etc... when doing polygonalization).

* Within Opticks create a corresponding interface class to bring in 
  the functionality, and treat it as an optional external library.
   
* Separated development promotes loose coupling 





EOU
}
opensource-dir(){ echo $(local-base)/env/strategy/strategy-opensource ; }
opensource-cd(){  cd $(opensource-dir); }
opensource-mate(){ mate $(opensource-dir) ; }
opensource-get(){
   local dir=$(dirname $(opensource-dir)) &&  mkdir -p $dir && cd $dir

}
