# === func-gen- : graphics/photonmap/photonmap fgp graphics/photonmap/photonmap.bash fgn photonmap fgh graphics/photonmap
photonmap-src(){      echo graphics/photonmap/photonmap.bash ; }
photonmap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(photonmap-src)} ; }
photonmap-vi(){       vi $(photonmap-source) ; }
photonmap-env(){      elocal- ; }
photonmap-usage(){ cat << EOU

https://code.google.com/p/275a-photonmap/source/checkout



EOU
}
photonmap-dir(){ echo $(local-base)/env/graphics/photonmap ; }
photonmap-cd(){  cd $(photonmap-dir); }
photonmap-mate(){ mate $(photonmap-dir) ; }
photonmap-get(){
   local dir=$(dirname $(photonmap-dir)) &&  mkdir -p $dir && cd $dir

   svn checkout http://275a-photonmap.googlecode.com/svn/trunk/ photonmap
}
