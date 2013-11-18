# === func-gen- : graphics/mesh/vcglib fgp graphics/mesh/vcglib.bash fgn vcglib fgh graphics/mesh
vcglib-src(){      echo graphics/meshlab/vcglib.bash ; }
vcglib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vcglib-src)} ; }
vcglib-vi(){       vi $(vcglib-source) ; }
vcglib-env(){      elocal- ; }
vcglib-usage(){ cat << EOU

VCGLIB
========

* http://vcg.isti.cnr.it/~cignoni/newvcglib/html/install.html
* http://vcg.sourceforge.net/index.php/Tutorial

EOU
}
vcglib-dir(){ echo $(local-base)/env/graphics/meshlab/vcglib ; }
vcglib-cd(){  cd $(vcglib-dir); }
vcglib-mate(){ mate $(vcglib-dir) ; }
vcglib-get(){
   local dir=$(dirname $(vcglib-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d vcglib ] && svn checkout svn://svn.code.sf.net/p/vcg/code/trunk/vcglib vcglib

}
