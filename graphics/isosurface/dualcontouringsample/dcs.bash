# === func-gen- : graphics/isosurface/dualcontouringsample/dcs fgp graphics/isosurface/dualcontouringsample/dcs.bash fgn dcs fgh graphics/isosurface/dualcontouringsample
dcs-src(){      echo graphics/isosurface/dualcontouringsample/dcs.bash ; }
dcs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dcs-src)} ; }
dcs-vi(){       vi $(dcs-source) ; }
dcs-env(){      elocal- ; }
dcs-usage(){ cat << EOU



http://ngildea.blogspot.co.uk/2014/11/implementing-dual-contouring.html

https://github.com/nickgildea/DualContouringSample



edgevmap
   relates edge index 0..11 to corner index 0..7
   foreach of the 12 cube edges, 4 for each of the 3 axes 


                 3:011    E3    7:111
                   +------------+
                   |            |  
                   |            |
                E5 |     +Z     | E7
                   |            | 
                   |            |
                   +------------+   
                 1:001   E1    5:101


         2:010   E2   6:110
         Y +------------+
           |            |  
           |            |
        E4 |            | E6
           |            | 
           |            |
           +------------+ X
         0:000  E0     4:100


   Last 4 edges difficult to draw


   0: 000     
   1: 001 
   2: 010 
   3: 011 
   4: 100  
   5: 101
   6: 110
   7: 111






EOU
}


dcs-dir(){    echo $HOME/DualContouringSample ; }
dcs-sdir(){   echo $HOME/DualContouringSample ; }
dcs-cd(){    cd $(dcs-dir)/$1 ; } 

dcs-bdir(){   echo $LOCAL_BASE/env/graphics/DualContouringSample.build ; }
dcs-prefix(){ echo $LOCAL_BASE/env/graphics/DualContouringSample ; }
dcs-bcd(){    cd $(dcs-bdir) ; } 


dcs-cd(){  cd $(dcs-dir); }
dcs-get()
{
   cd $HOME
   #local url=https://github.com/nickgildea/DualContouringSample
   local url=https://github.com/simoncblyth/DualContouringSample
   [ ! -d $(basename $url) ] && git clone $url
}


dcs-cmake()
{
    local bdir=$(dcs-bdir)

    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    dcs-bcd   
    opticks-
    # notice no glm precursor here, it comes it a CMake level 

    cmake \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(dcs-prefix) \
       $* \
       $(dcs-sdir)
}

dcs-export()
{
   local libdir=$(dcs-prefix)/lib
   [ "${PATH/$libdir}" == "$PATH"  ] && export PATH=$libdir:$PATH || echo $msg libdir $libdir already in PATH
}

dcs--()
{
   local msg="$FUNCNAME : "
   local iwd=$PWD
   dcs-bcd

   cmake --build . --target ${1:-install}

   cd $iwd
}




