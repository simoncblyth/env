devil-source(){   echo ${BASH_SOURCE} ; }
devil-edir(){ echo $(dirname $(devil-source)) ; }
devil-ecd(){  cd $(devil-edir); }
devil-dir(){  echo $LOCAL_BASE/env/graphics/image/DevIL ; }
devil-cd(){   cd $(devil-dir); }
devil-vi(){   vi $(devil-source) ; }
devil-env(){  elocal- ; }
devil-usage(){ cat << EOU

DevIL : Developer's Image Library
===================================

DevIL is a cross-platform image library utilizing a simple syntax to load, 
save, convert, manipulate, filter, and display a variety of images with ease. 
It is highly portable and has been ported to several platforms. 
 
* http://openil.sourceforge.net/
* https://github.com/DentonW/DevIL

 
Externals
----------

::

    -- Could NOT find cppunit (missing: CPPUNIT_LIBRARY CPPUNIT_INCLUDE_DIR) 
    -- UnitTest disabled, cppunit wasn't found!


CMake FindDevIL.cmake is rather useless
---------------------------------------------

Message from OptiX Advanced Samples::

    DevIL image library not found. 
    Please set IL_LIBRARIES, ILU_LIBRARIES, ILUT_LIBRARIES, and IL_INCLUDE_DIR to build OptiX introduction samples 07 to 10

::

    blyth@localhost optixIntroduction]$ find /usr/share/cmake3/ -name FindDevIL.cmake
    /usr/share/cmake3/Modules/FindDevIL.cmake






EOU
}
devil-get(){
   local dir=$(dirname $(devil-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d DevIL ] && git clone git@github.com:simoncblyth/DevIL.git

}

devil-sdir(){ echo $(devil-dir)/DevIL ; }
devil-bdir(){ echo $(devil-dir).build ; }
devil-idir(){ echo $(devil-dir)/local ; }

devil-install-dir(){ echo $(devil-idir) ; }

devil-scd(){ cd $(devil-sdir) ; }
devil-bcd(){ cd $(devil-bdir) ; }

devil-cmake(){
   local iwd=$PWD
   local bdir=$(devil-bdir)
   local sdir=$(devil-sdir)
   local idir=$(devil-idir)
   mkdir -p $idir

   mkdir -p $bdir && cd $bdir
   cmake $sdir \
       -DCMAKE_INSTALL_PREFIX=$idir

   cd $iwd
}

devil-make()
{
   local iwd=$PWD
   devil-bcd 
   make $* 
   cd $iwd
}

devil--()
{
   devil-get
   devil-cmake
   devil-make
   devil-make install
}

devil-lib(){ echo $(devil-install-dir)/lib/lib${1}.so ; }
devil-include-dir(){ echo $(devil-install-dir)/include ; }
devil-info(){ cat << EOI

   devil-lib IL    : $(devil-lib IL)
   devil-lib ILU   : $(devil-lib ILU)
   devil-lib ILUT  : $(devil-lib ILUT)

   devil-include-dir : $(devil-include-dir)

EOI
}

devil-libs(){ cat << EOL
$(devil-lib IL)
$(devil-lib ILU)
$(devil-lib ILUT)
EOL
}


