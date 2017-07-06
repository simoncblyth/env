# === func-gen- : graphics/gts/gts fgp graphics/gts/gts.bash fgn gts fgh graphics/gts
gts-src(){      echo graphics/gts/gts.bash ; }
gts-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gts-src)} ; }
gts-vi(){       vi $(gts-source) ; }
gts-env(){      elocal- ; }
gts-usage(){ cat << EOU

GTS : GNU Triangulated Surface Library
=========================================

* http://gts.sourceforge.net
* http://gts.sourceforge.net/gallery.html

GNU Triangulated Surface Library. It is an Open Source Free Software Library
intended to provide a set of useful functions to deal with 3D surfaces meshed
with interconnected triangles. The source code is available free of charge
under the Free Software LGPL license.







EOU
}


gts-url(){ echo http://gts.sourceforge.net/tarballs/gts-snapshot-121130.tar.gz ; }
gts-nam(){ local dst=$(basename $(gts-url)) ; echo ${dst/.tar.gz} ; }

gts-dir(){ echo $(local-base)/env/graphics/gts/$(gts-nam) ; }
gts-cd(){  cd $(gts-dir); }
gts-get(){
   local dir=$(dirname $(gts-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(gts-url)
   local nam=$(gts-nam)
   local dst=${nam}.tar.gz

   [ ! -f "$dst" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $dst  
   

}


gts-bbtree(){ vi $(gts-dir)/src/bbtree.c $(gts-dir)/src/gts.h; }
