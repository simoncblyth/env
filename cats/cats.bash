cats-env(){ echo -n ; }
cats-vi(){ vi $BASH_SOURCE ; }
cats-usage(){ cat << EOU
cats.bash
===========

~/Downloads/alexei.txt 

EOU
}

cats-base(){ echo ${CATS_BASE:-/data/blyth/cats} ; }
cats-clhep-url(){  echo https://proj-clhep.web.cern.ch/proj-clhep/dist1/clhep-2.4.6.2.tgz ; }
cats-geant4-url(){ echo https://geant4-data.web.cern.ch/releases/geant4-v11.1.2.tar.gz ; } 

cats-init(){
    local base=$(cats-base)
    mkdir -p $base && cd $base
}

cats-curl(){
    local url=$1    # distribution url 
    local dist=$2   # distribution filename
    local edir=$3   # expanded dir 

    if [ ! -d "$edir" ]; then  
        [ ! -f "$dist" ] && curl -L -O $url 
        [ -f "$dist" ] && echo $FUNCNAME : dist $dist already downloaded from url $url  
        tar xzvf $dist 
    else
        echo $FUNCNAME : dist already expanded into edir $edir
    fi
}

cats-clhep(){
    cats-init
    local url=$($FUNCNAME-url)
    local dist=$(basename $url)
    local edir=${dist/.tgz}
    edir=${edir/clhep-}

    cats-curl $url $dist $edir

    cd $edir
    local bdir=CLHEP-build
    local idir=CLHEP-install
    mkdir -p $bdir && cd $bdir

    # -GNinja
    cmake \
        -DCMAKE_INSTALL_PREFIX=$idir \ 
        -DCMAKE_BUILD_TYPE=Release \
        -DCLHEP_BUILD_CXXSTD=-std=c++17 \
        ../CLHEP
    make
    make install
}


cats-geant4(){
    cats-init
    local url=$($FUNCNAME-url)
    local dist=$(basename $url)
    local edir=${dist/.tar.gz}

    cats-curl $url $dist $edir

    local bdir=${edir}-build
    local idir=${edir}-install

    mkdir -p $bdir && cd $bdir

    # -GNinja 
    cmake  \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=../$idir \
        -DGEANT4_BUILD_BUILTIN_BACKTRACE=OFF \
        -DGEANT4_BUILD_VERBOSE_CODE=OFF \
        -DGEANT4_INSTALL_DATA=ON \
        -DGEANT4_USE_SYSTEM_CLHEP=ON \
        -DGEANT4_USE_GDML=ON \
        -DGEANT4_USE_SYSTEM_EXPAT=ON \
        -DGEANT4_USE_SYSTEM_ZLIB=ON \
        -DGEANT4_USE_QT=ON \
        -DGEANT4_BUILD_MULTITHREADED=OFF \
        -DGEANT4_USE_OPENGL_X11=ON \
         ../$edir

    make
    make install
}

cats-root(){
    cats-init

    local ver=6_28_04
    
    git clone --branch latest-stable https://github.com/root-project/root.git root_${ver}_src
    mkdir root_${ver}_build
    cd root_${ver}_build

    cmake -GNinja \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_INSTALL_PREFIX=../root_${ver}-install \
          -Droot7=ON \
          -Dxrootd=OFF \
          ../root_${ver}_src/

    cmake --build . --target install
    #source ../root_${ver}-install/bin/thisroot.sh
    #root

 
}


cats-cats(){

    cats-init

    git clone https://github.com/hanswenzel/CaTS.git
    cd CaTS/

    #source (path to opticks WORK_DIR)/setup_opticks.sh
    #cd ../
    #mkdir CaTS-build
    #cd CaTS-build
    #-GNinja \

    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_G4CXOPTICKS=ON \
      -DCMAKE_PREFIX_PATH="${LOCAL_BASE}/opticks/externals;${LOCAL_BASE}/opticks" \
      -DOPTICKS_PREFIX=${LOCAL_BASE}/opticks \
      -DCMAKE_MODULE_PATH=${OPTICKS_HOME}/cmake/Modules \
      -DCMAKE_INSTALL_PREFIX=../CaTS-install \
      ../CaTS

    make install
    cd ../CaTS-install/bin
}




