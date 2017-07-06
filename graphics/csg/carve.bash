# === func-gen- : graphics/csg/carve fgp graphics/csg/carve.bash fgn carve fgh graphics/csg
carve-src(){      echo graphics/csg/carve.bash ; }
carve-source(){   echo ${BASH_SOURCE:-$(env-home)/$(carve-src)} ; }
carve-vi(){       vi $(carve-source) ; }
carve-env(){      elocal- ; }
carve-usage(){ cat << EOU


Carve (GPL) CSG lib used by Blender
=======================================

* releases looked ancient so downloaded source archive

Last Google Code Commit 
--------------------------

Tobias Sargeant Jun 24, 2014    9a85d733a43d    Re-license Carve 2.0, allowing selection of GPL2 or GPL3.

* https://code.google.com/archive/p/carve/
* https://code.google.com/archive/p/carve/source/default/commits

* https://github.com/VTREEM/Carve

Google code is no more...

hg clone https://code.google.com/p/carve

Related
---------

* https://stackoverflow.com/questions/16102029/carve-csg-library
* https://github.com/jamesgregson/pyPolyCSG

The Below Looks to be in the lead
------------------------------------

* https://github.com/qnzhou/carve
* https://github.com/qnzhou/carve/commits/master




EOU
}
carve-dir(){ echo $(local-base)/env/graphics/csg/carve ; }
carve-cd(){  cd $(carve-dir); }

#carve-url(){ echo https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/carve/carve-1.4.0.tgz ; }
carve-url(){ echo https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/carve/source-archive.zip ; }

carve-get(){
   local dir=$(dirname $(carve-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(carve-url)
   local dst=$(basename $url)
   [ ! -f "$dst" ] && curl -L -O $url
   [ ! -d carve ] && unzip $dst 

}
