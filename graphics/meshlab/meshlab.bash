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
#. $(meshlab-fold)/meshlab contains my github customized Meshlab clone 

Installation
--------------

#. build externals::

    meshlab-get
    meshlab-external
    make                  # qmake must be in PATH

#. build meshlab::

    meshlab-cd
    meshlab-qmake
    meshlab-make

#. run::

    meshlab--    # default is to load mesh corresponding to 3155___2 


DBUS Requirements
-------------------

For DBUS communication to work need the DBUS session and system daemons
to be running prior to launching the forked MeshLab

TODO
------

#. check on DBUS prior to MeshLab launch ? 
#. migrate MeshLab notes/docs on enhancements out of env and into:

   * bitbucket wiki repo ? 
   * Hmm not keen on:

     * lock-in aspect
     * having a separate repo just for a few docs
     * dealing with doc system inferior to Sphinx  (no toc)

#. So: add a Sphinx style docs directory with toctree to the forked MeshLab 


Deployment Thoughts
-----------------------

Need to create a populated app bundle with the Qt frameworks in place 
in order for the customized MeshLab (faster DAE loading would be 
very convenient to have) to work on general machines without
the macports installed Qt frameworks at system level.

Also will generally be no DBUS operational, so need to consider how
to degrade on machines without DBUS, or without the DBUS daemon running.

* http://qt-project.org/doc/qt-4.8/deployment-mac.html



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

*meshlab---*
         launch the meshlab GUI. 

*meshlab-- <volspec>*
         launch the meshlab GUI and load mesh corresponding to volspec. Eg 3155___2 


*meshlab-collada-make*
         separate build collada plugin
*meshlab-collada-install*
         install into separate build plugins folder
*meshlab-collada-promote*
         promote from separate build plugins folder into official plugins folder


DAE FILES
----------

Full files are scp from N::

    simon:daeserver blyth$ pwd
    /usr/local/env/geant4/geometry/daeserver
    simon:daeserver blyth$ scp N:/data1/env/local/env/geant4/geometry/daeserver/DVGX_20131121-2053_g4_00.dae .
    simon:daeserver blyth$ scp N:/data1/env/local/env/geant4/geometry/daeserver/VDGX_20131121-2043_g4_00.dae .

Sub geometries can be automatically pulled off the daeserver if not already available, 
by launching meshlab with the appropriate arg eg::

    meshlab-- 3155___2
    meshlab-- DVGX_20131121-2053_g4_00
    meshlab-- VDGX_20131121-2043_g4_00


EXTERNAL NAVIGATION
---------------------

Symptom of DBUS server not running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:bitbucket blyth$ meshlab-v "http://localhost/dae/tree/0.html?c=0.001&a=0,0,-1&fov=60"
    Dynamic session lookup supported but failed: launchd did not provide a socket path, verify that org.freedesktop.dbus-session.plist is loaded!
    Could not connect to D-Bus server: org.freedesktop.DBus.Error.NoMemory: Not enough memory


Perspective Views
~~~~~~~~~~~~~~~~~~~

Perspective view with LookAt control::

     meshlab-v "http://localhost/dae/tree/0.html?c=0.001&a=0,0,-1&fov=60"
     meshlab-v "http://localhost/dae/tree/0.html?c=0.001&a=0,-1,0&fov=60"


#. *c=0.001* indicates camera close to center of volume bbox at 0.001,0.001,0.001 
   (avoid camera position 0,0,0 it causes issues)
#. *a=0,0,-1* looking down -Z 
          

TODO: 

#. switching to orthographic with fov=5 causes everything to be clipped



Orthographic Axis Views
~~~~~~~~~~~~~~~~~~~~~~~~~~

Orthographic view using **orth** OR **o** param with values l/r/t/b/k/f OR Left/Right/Top/Bottom/Back/Front
relative to the bounding box of the identified volume::

    meshlab-v "http://localhost/dae/tree/10.html?orth=Left"
    meshlab-v "http://localhost/dae/tree/10.html?orth=Top"
    meshlab-v "http://localhost/dae/tree/10.html?orth=Back"
    meshlab-v "http://localhost/dae/tree/10.html?orth=Front"
    meshlab-v "http://localhost/dae/tree/10.html?orth=Left"
    meshlab-v "http://localhost/dae/tree/10.html?o=l"
    meshlab-v "http://localhost/dae/tree/10.html?o=r"
    meshlab-v "http://localhost/dae/tree/10.html?o=t"
    meshlab-v "http://localhost/dae/tree/10.html?o=b"
    meshlab-v "http://localhost/dae/tree/10.html?o=k"
    meshlab-v "http://localhost/dae/tree/10.html?o=f"
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
meshlab-wiki-get(){
   local dir=$(meshlab-fold) &&  mkdir -p $dir && cd $dir
   [   -d meshlab_wiki ] && echo meshlab_wiki exists already && return 1
   git clone http://bitbucket.org/scb-/meshlab/wiki  meshlab_wiki
}
meshlab-wiki-cd(){ cd $(meshlab-fold)/meshlab_wiki ;  }

meshlab-vcgdir(){ echo $(dirname $(dirname $(meshlab-dir)))/vcglib ; }

meshlab-env(){      
   elocal- 
   qt4- 

  # export MESHLAB_DIR=$(meshlab-dir)
  # export MESHLAB_VCGDIR=$(meshlab-vcgdir)
}


meshlab-dae-dir(){  echo $LOCAL_BASE/env/geant4/geometry/daeserver ; }
meshlab-dae-path(){ echo $(meshlab-dae-dir)/${1}.dae ; }
meshlab-dae-url(){  echo http://belle7.nuu.edu.tw/dae/tree/${1}.dae ; }
meshlab-dae-ls(){   ls -l $(meshlab-dae-dir) ; }
meshlab-dae-cd(){   cd $(meshlab-dae-dir) ; }
meshlab-dae-get(){
   local path=$(meshlab-dae-path $1)
   local url=$(meshlab-dae-url $1)
   local dir=$(dirname $path)
   mkdir -p $dir
   local cmd="curl -s $url -o $path "
   [ ! -f "$path" ] && echo $cmd && eval $cmd
   ls -l $path
   du -h $path
   echo ----------------------- HEAD
   head $path
   echo ----------------------- TAIL
   tail $path
}

meshlab--(){
   local base=${1:-3155___2}
   local path=$(meshlab-dae-path $base)
   meshlab-dae-get $base
   meshlab--- $path
}
meshlab---(){
   meshlab-cd distrib/meshlab.app/Contents/MacOS   # attemping to launch from elsewhere fails to load plugins
   ./meshlab $*
}




meshlab-v-url(){ echo "http://localhost/dae/tree/${1:-0}.html?fov=30&cam=1&a=0.1,0.1,0.1&up=0,0,1" ; }
meshlab-v(){  qdbus com.meshlab.navigator / SayHelloThere "$1" ; }
meshlab-vv(){ meshlab-v "$(meshlab-v-url $1)" ; }

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
   cd $(meshlab-dir)/meshlabplugins/io_collada
   [ -f Makefile ] && make distclean
   env | grep MESHLAB
   qmake
   make
}
meshlab-collada-install(){
   cd $(meshlab-dir)/meshlabplugins/io_collada
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
meshlab-distrib-dir(){ echo $(meshlab-dir)/distrib ; }

meshlab-server-make(){
   cd $(meshlab-dir)/meshlabserver
   rm -rf ../distrib/meshlab.app/Contents/MacOS/meshlabserver
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
