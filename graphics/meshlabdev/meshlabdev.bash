# === func-gen- : graphics/meshlabdev/meshlabdev fgp graphics/meshlabdev/meshlabdev.bash fgn meshlabdev fgh graphics/meshlabdev
meshlabdev-src(){      echo graphics/meshlabdev/meshlabdev.bash ; }
meshlabdev-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshlabdev-src)} ; }
meshlabdev-vi(){       vi $(meshlabdev-source) ; }
meshlabdev-env(){      elocal- ; }
meshlabdev-usage(){ cat << EOU

MESHLABDEV
============

Add hoc approach to meshlab dev over in meshlab- is 
getting unmanageable.  


EOU
}
meshlabdev-dir(){ echo $(local-base)/env/graphics/meshlabdev ; }
meshlabdev-cd(){  cd $(meshlabdev-dir); }
meshlabdev-mate(){ mate $(meshlabdev-dir) ; }
meshlabdev-get(){
   local dir=$(meshlabdev-dir) &&  mkdir -p $dir && cd $dir

   git svn clone http://svn.code.sf.net/p/meshlab/code/trunk/  meshlab_trunk
   git svn clone http://svn.code.sf.net/p/vcg/code/trunk/ vcglib_trunk

}
