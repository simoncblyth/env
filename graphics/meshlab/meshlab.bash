# === func-gen- : graphics/mesh/meshlab fgp graphics/mesh/meshlab.bash fgn meshlab fgh graphics/mesh
meshlab-src(){      echo graphics/meshlab/meshlab.bash ; }
meshlab-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshlab-src)} ; }
meshlab-vi(){       vi $(meshlab-source) ; }
meshlab-usage(){ cat << EOU

Meshlab
========


FUNCTIONS
----------

*meshlab-external*
         qmake the Makefile for externals

*meshlab-config*
         emit name of qmake project file that defines which plugins to build together with core meshlab

*meshlab-qmake*
         qmake the Makefile for configured meshlab build, and kludge fixes the generated Makefiles

*meshlab-make*
         invoke the Makefiles

*meshlab-build*
         do above qmake and make functions


EOU
}
meshlab-dir(){ echo $(local-base)/env/graphics/meshlab/meshlab/src ; }
meshlab-cd(){  cd $(meshlab-dir)/$1 ; }
meshlab-scd(){  cd $(env-home)/graphics/meshlab/$1 ; }
meshlab-mate(){ mate $(meshlab-dir) ; }
meshlab-get(){
   local dir=$(dirname $(dirname $(meshlab-dir))) &&  mkdir -p $dir && cd $dir

   local tar=MeshLabSrc_AllInc_v132.tar
   echo  SF DOWNLOADING IS BROKEN : HAVE TO DO MANUALLY : mv ~/Downloads/$tar . 
}

meshlab-vcgdir(){ echo $(dirname $(dirname $(meshlab-dir)))/vcglib ; }

meshlab-env(){      
   elocal- 
   qt4- 

   export MESHLAB_DIR=$(meshlab-dir)
   export MESHLAB_VCGDIR=$(meshlab-vcgdir)
}
meshlab-launch(){
   meshlab-cd distrib/meshlab.app/Contents/MacOS
   ./meshlab
}

meshlab-find(){ find $(meshlab-dir) -name '*.cpp' -exec grep -H $1 {} \; ;  }
meshlab-plugins-dir(){  echo $(meshlab-dir)/distrib/plugins ; }
meshlab-plugins-ls(){   ls -l $(meshlab-plugins-dir) ; }
meshlab-plugins-cd(){   cd $(meshlab-plugins-dir) ; }

meshlab-config(){ 
   #echo  meshlab_full.pro 
   echo  meshlab_mini.pro 
}

meshlab-external(){
   type $FUNCNAME
   meshlab-cd external
   qmake -recursive external.pro
}
meshlab-qmake(){
   type $FUNCNAME
   meshlab-cd 
   qmake -recursive $(meshlab-config)
   qt4-kludge
}
meshlab-make(){
   type $FUNCNAME
   meshlab-cd 
   make
}

meshlab-build(){
   meshlab-qmake && meshlab-make
}
