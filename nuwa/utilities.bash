# === func-gen- : nuwa/utilities fgp nuwa/utilities.bash fgn utilities fgh nuwa
utilities-src(){      echo nuwa/utilities.bash ; }
utilities-source(){   echo ${BASH_SOURCE:-$(env-home)/$(utilities-src)} ; }
utilities-vi(){       vi $(utilities-source) ; }
utilities-env(){      elocal- ; }

utilities-usage(){ cat << EOU

NuWa Utilities
=================

Objective
----------

Collective handling of the utilities

Players
--------

*G4DAEChroma*

   * http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/Utilities/G4DAEChroma
   * env/chroma/G4DAEChroma
   * depends on zmq?, ZMQRoot, Chroma
   * huh no Geant4 in requirements
   * see *gdc-*

*ChromaZMQRootTest*

   * http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/Utilities/ChromaZMQRootTest
   * env/chroma/ChromaZMQRootTest
   * depends on zmq, ZMQRoot, Chroma
   * see *czrt-* 

*Chroma* 
   (de)serialization class for photon transport over networks

   * http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/Utilities/Chroma
   * env/chroma/ChromaPhotonList
   * tricky rootcinting needed
   * depends on ROOT, Geant4 
   * see *cpl-* for cmake based build

*ZMQRoot*

   * http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/Utilities/ZMQRoot
   * env/zmqroot
   * depends on zmq, ROOT
   * see *zmqroot-* 




EOU
}
utilities-dir(){ echo $(local-base)/env/nuwa/utilities ; }
utilities-cd(){  cd $(utilities-dir); }
utilities-mate(){ mate $(utilities-dir) ; } 
utilities-get(){
   local dir=$(dirname $(utilities-dir)) &&  mkdir -p $dir && cd $dir

}


utilities-udir(){
   local name=${1:-G4DAEChroma}
   echo $DYB/NuWa-trunk/dybgaudi/Utilities/$name
}

utilities-names(){ cat << EON
G4DAEChroma
ChromaZMQRootTest
Chroma
ZMQRoot
EON
}

utilities-ls(){
  local name
  utilities-names | while read name ; do
     local dir=$(utilities-udir $name)
     local cmd="ls $dir"
     echo $cmd
     eval $cmd
  done

}



