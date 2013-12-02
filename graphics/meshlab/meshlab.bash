# === func-gen- : graphics/mesh/meshlab fgp graphics/mesh/meshlab.bash fgn meshlab fgh graphics/mesh
meshlab-src(){      echo graphics/meshlab/meshlab.bash ; }
meshlab-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshlab-src)} ; }
meshlab-vi(){       vi $(meshlab-source) ; }
meshlab-usage(){ cat << EOU

Meshlab
========

Note there are two areas with meshlab sources and builds:

#. Standard meshlab in $(meshlab-dir) 
#. Optimized meshlab collada plugin, and meshlabserver inside the working copy $(env-home)/graphics/meshlab 

Instllation::

    meshlab-get
    meshlab-external
    make                  # qmake must be in PATH


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

*meshlab--*
         launch the meshlab GUI. Optionally load a mesh directly::

             meshlab-- $(local-base)/env/geant4/geometry/gdml/3199.dae


*meshlab-collada-make*
         separate build collada plugin
*meshlab-collada-install*
         install into separate build plugins folder
*meshlab-collada-promote*
         promote from separate build plugins folder into official plugins folder






EOU
}
meshlab-dir(){ echo $(local-base)/env/graphics/meshlab/meshlab/src ; }
meshlab-fold(){ echo $(dirname $(dirname $(dirname $(meshlab-dir)))) ;}
meshlab-cd(){  cd $(meshlab-dir)/$1 ; }
meshlab-scd(){  cd $(env-home)/graphics/meshlab/$1 ; }
meshlab-mate(){ mate $(meshlab-dir) ; }


meshlab-get-original(){
   local dir=$(meshlab-fold) &&  mkdir -p $dir && cd $dir
   echo SF IS SUCH A PAIN : IT BEATS ME HOW IT MANAGES TO SURVIVE
   pwd
   local tgz=MeshLabSrc_AllInc_v132.tgz
   local nam=${tgz/.tgz}
   [ ! -f "$tgz" ] && curl -L "http://sourceforge.net/projects/meshlab/files/meshlab/MeshLab%20v1.3.2/MeshLabSrc_AllInc_v132.tgz/download?use_mirror=nchc" -o $tgz  
   #   [ ! -d "meshlab" ] && echo CREATING CONTAINER DIR TO HANDLE EXPLODING TGZ && mkdir meshlab && ( cd meshlab && tar zxvf ../$tgz )
    [ ! -d "meshlab" ] && echo WARNING EXPLODING TGZ && tar zxvf $tgz 
}

meshlab-url(){ echo https://bitbucket.org/scb-/meshlab.git ; }
meshlab-clone(){
   local url=$(meshlab-url)
   case $NODE_TAG in 
      N)  gitsrc- && which git && git --version && GIT_SSL_NO_VERIFY=true git clone $url ;;
      *)  git clone $url ;;
   esac
}
meshlab-get(){
   local dir=$(meshlab-fold) &&  mkdir -p $dir && cd $dir
   [   -d meshlab ] && echo meshlab exists already && return 1
   [ ! -d meshlab ] && meshlab-clone
}


meshlab-vcgdir(){ echo $(dirname $(dirname $(meshlab-dir)))/vcglib ; }

meshlab-env(){      
   elocal- 
   qt4- 

   export MESHLAB_DIR=$(meshlab-dir)
   export MESHLAB_VCGDIR=$(meshlab-vcgdir)
}
meshlab--(){
   # attemping to launch from elsewhere fails to load plugins
   meshlab-cd distrib/meshlab.app/Contents/MacOS
   ./meshlab $*
}

meshlab-q(){ meshlab-- /usr/local/env/geant4/geometry/gdml/${1:-3199}.dae ; }
meshlab-v-url(){ echo "http://localhost/dae/tree/${1:-0}.html?fov=30&cam=1&a=0.1,0.1,0.1&up=0,0,1" ; }
meshlab-v(){ qdbus com.meshlab.navigator / SayHelloThere "$(meshlab-v-url $1)" ; }

meshlab--server(){
   # attemping to launch from elsewhere fails to load plugins
   meshlab-cd distrib/meshlab.app/Contents/MacOS
   ./meshlabserver $*
}


meshlab-dae(){ echo  /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae ; }
meshlab--server-test(){
   meshlab--server -i $(meshlab-dae)
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

   case $NODE_TAG in 
      G) qt4-kludge ;;
   esac
}
meshlab-make(){
   type $FUNCNAME
   meshlab-cd 
   make
}

meshlab-build(){
   meshlab-qmake && meshlab-make
}



########### BELOW FOR DEVELOPMENT TESTING OF CUSTOMOMIZED MESHLAB ########################

meshlab-collada-make(){
   cd $(env-home)/graphics/meshlab/meshlabplugins/io_collada
   [ -f Makefile ] && make distclean
   env | grep MESHLAB
   qmake
   make
}
meshlab-collada-install(){
   cd $(env-home)/graphics/meshlab/meshlabplugins/io_collada
   local dest=../../distrib/plugins/
   mkdir -p $dest
   local plug=libio_collada.dylib
   local target=$dest/$plug
   rm -rf $target
   [ -f $plug ] && cp $plug $target
}
meshlab-collada-ls(){
   echo $(meshlab-distrib-dir)/plugins/
   ls -l $(meshlab-distrib-dir)/plugins/
}
meshlab-distrib-dir(){ echo $(env-home)/graphics/meshlab/distrib ; }
meshlab-collada-promote(){
   echo promoting to the offical meshlab build plugins
   cp $(meshlab-distrib-dir)/plugins/libio_collada.dylib $(meshlab-dir)/distrib/plugins/
}

meshlab-server-make(){
   cd $(env-home)/graphics/meshlab/meshlabserver
   rm -rf ../distrib/meshlab.app 
   [ -f Makefile ] && make distclean
   env | grep MESHLAB
   qmake
   make
}
meshlab-server-install(){
   type $FUNCNAME
   local app=$(meshlab-server-app)
   cp ${MESHLAB_DIR}/common/*.dylib $app/Contents/MacOS/ 
   meshlab-collada-install 
}
meshlab-server-app(){ echo $(meshlab-distrib-dir)/meshlab.app/ ; }
meshlab-server-ls(){
    ls -Ralst $(meshlab-distrib-dir) 
}


meshlab-server-test(){
    type $FUNCNAME
    local app=$(meshlab-server-app)
    cd $app/Contents/MacOS
    ./meshlabserver -i /usr/local/env/geant4/geometry/gdml/VDGX_20131121-1957/g4_00.dae
}
