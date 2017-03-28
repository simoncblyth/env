# === func-gen- : graphics/mortonlib/mortonlib fgp graphics/mortonlib/mortonlib.bash fgn mortonlib fgh graphics/mortonlib
mortonlib-src(){      echo graphics/mortonlib/mortonlib.bash ; }
mortonlib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mortonlib-src)} ; }
mortonlib-vi(){       vi $(mortonlib-source) ; }
mortonlib-env(){      elocal- ; }
mortonlib-usage(){ cat << EOU

MortonLib (MIT)
================

* https://github.com/aavenel/mortonlib

Alternates
-----------

* https://github.com/Forceflow/libmorton (GPL)

About BMI2 
--------------

* http://www.forceflow.be/2016/11/25/using-the-bmi2-instruction-set-to-encode-decode-morton-codes/


MortonLib Tests
----------------


Classic corresponds to Grid2d::

     48   Grid2d<gridType> g = Grid2d<gridType>(gridsize);
     49   MortonGrid2d<gridType> gm = MortonGrid2d<gridType>(gridsize);

Classic is just standard array indexing, it beats Morton in all tests::

    delta:mortonlib.build blyth$ tests/morton_test 
    Classic 2d grid get() linear : 39.4384ms
    Morton  2d grid get() linear : 188.224ms
    Classic 2d grid get() random : 91.9447ms
    Morton  2d grid get() random : 345.842ms
    Classic 2d grid get() linear non cache friendly : 54.589ms
    Morton  2d grid get() linear non cache friendly : 224.642ms
    Classic 2d grid get() random + 4 neighbors : 339.084ms
    Morton  2d grid get() random + 4 neighbors : 688.438ms
    Classic 2d grid get() random + 8 neighbors : 838.128ms
    Morton  2d grid get() random + 8 neighbors A : 2522.33ms
    Morton  2d grid get() random + 8 neighbors B : 1204.43ms
    Morton  2d grid get() random + 8 neighbors B'  : 1161.05ms
    Classic 3d grid get() linear : 43.4796ms
    Morton  3d grid get() linear : 168.055ms
    Classic 3d grid get() random : 398.274ms
    Morton  3d grid get() random : 686.88ms
    Classic 3d grid get() linear non cache friendly : 145.267ms
    Morton  3d grid get() linear non cache friendly : 236.637ms
    Classic 3d grid get() random + 6 neighbors : 1115.86ms
    Morton  3d grid get() random + 6 neighbors : 1404.17ms
    Classic 3d grid get() random + 26 neighbors : 3027.91ms
    Morton  3d grid get() random + 26 neighbors A : 8309.19ms
    Morton  3d grid get() random + 26 neighbors B  : 3091.4ms
    delta:mortonlib.build blyth$ 





EOU
}
mortonlib-dir(){ echo $(local-base)/env/graphics/mortonlib ; }
mortonlib-cd(){  cd $(mortonlib-dir); }
mortonlib-mate(){ mate $(mortonlib-dir) ; }
mortonlib-get(){
   local dir=$(dirname $(mortonlib-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d mortonlib ] && git clone https://github.com/aavenel/mortonlib
}

mortonlib-bdir(){   echo $(mortonlib-dir).build ; }
mortonlib-prefix(){ echo $(mortonlib-dir).install ; }
mortonlib-bcd(){  cd $(mortonlib-bdir) ; }


mortonlib-cmake(){
  local iwd=$PWD
  local bdir=$(mortonlib-bdir)
  mkdir -p $bdir

  mortonlib-bcd

  cmake $(mortonlib-dir) \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$(mortonlib-prefix)

  cd $iwd
}

mortonlib-make(){
  local iwd=$PWD
  mortonlib-bcd

  make

  cd $iwd
}

mortonlib-extract-(){ cat << EOL
LICENSE.txt
include/morton3d.h
include/morton2d.h
EOL
}

mortonlib-extract()
{
   local targetdir=${1:-${OPTICKS_HOME}/opticksnpy}
   [ ! -d "$targetdir" ] && echo $msg targetdir must exist && return 

   local destdir=$targetdir/mortonlib
   mkdir -p $destdir 

   mortonlib-cd 

   local rel
   mortonlib-extract- | while read rel ; do
       echo cp $rel $destdir/$(basename $rel)
   done   
}


