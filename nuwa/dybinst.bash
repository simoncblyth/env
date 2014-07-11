# === func-gen- : nuwa/dybinst fgp nuwa/dybinst.bash fgn dybinst fgh nuwa
dybinst-src(){      echo nuwa/dybinst.bash ; }
dybinst-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dybinst-src)} ; }
dybinst-vi(){       vi $(dybinst-source) ; }
dybinst-env(){      elocal- ; }
dybinst-usage(){ cat << EOU

BNL tarFiles Outage
----------------------

URLs like http://dayabay.bnl.gov/software/offline/tarFiles/aida-3.2.1.tar.gz 
Download to $DYB/external/tarFiles/aida-3.2.1.tar.gz

Publish cms01(belle7 not accessible yet again) tarballs at BNL mimic layout:

* http://cms01.phys.ntu.edu.tw/software/offline/tarFiles/
* see nginx- for setup 

::

    [blyth@ntugrid5 NuWa-trunk]$ find . -name requirements -exec grep -H dayabay.bnl.gov {} \;
    ./lcgcmt/LCG_Builders/LCG_BuildPolicy/cmt/requirements:    dayabay               "http://dayabay.bnl.gov/software/offline/tarFiles" 

    set LCG_tarurl                       "http://service-spi.web.cern.ch/service-spi/external/tarFiles" \
        dayabay                          "http://dayabay.bnl.gov/software/offline/tarFiles"



ntugrid5 dybinst attempt
--------------------------

* http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/ThhoDybinst
* https://wiki.bnl.gov/dayabay/index.php?title=RACF_Upgrade_to_Scientific_Linux_6.2


gaudi : missing libuuid-devel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    c6-gcc44-dbg/root/include"   -I"/home/blyth/local/env/dyb/NuWa-trunk/lcgcmt/LCG_Interfaces/ROOT/src" -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/ROOT/5.26.00e_python2.7/x86_64-slc6-gcc44-dbg/root/include"   -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/Boost/1.38.0_python2.7/x86_64-slc6-gcc44-dbg/include/boost-1_38"   -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/Python/2.7/x86_64-slc6-gcc44-dbg/include/python`echo 2.7 | cut -d. -f1,2`"     -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/AIDA/3.2.1/share/src/cpp"   -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/uuid/1.38/x86_64-slc6-gcc44-dbg/include"   -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/XercesC/2.8.0/x86_64-slc6-gcc44-dbg/include"   -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/clhep/2.0.4.2/x86_64-slc6-gcc44-dbg/include"   -I"/home/blyth/local/env/dyb/NuWa-trunk/../external/pcre/4.4/x86_64-slc6-gcc44-dbg/include"     -I"/home/blyth/local/env/dyb/NuWa-trunk/lcgcmt/LCG_SettingsCompat/src"   -I"/home/blyth/local/env/dyb/NuWa-trunk/lcgcmt/Documentation/Doxygenated/src"        -I../src/component ../src/component/XMLFileCatalog.cpp
    In file included from /usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/ext/hash_map:60,
                     from /home/blyth/local/env/dyb/NuWa-trunk/gaudi/InstallArea/include/GaudiKernel/HashMap.h:17,
                     from /home/blyth/local/env/dyb/NuWa-trunk/gaudi/InstallArea/include/GaudiKernel/SerializeSTL.h:22,
                     from /home/blyth/local/env/dyb/NuWa-trunk/gaudi/InstallArea/include/GaudiKernel/MsgStream.h:7,
                     from ../src/component/XMLFileCatalog.cpp:14:
    /usr/lib/gcc/x86_64-redhat-linux/4.4.7/../../../../include/c++/4.4.7/backward/backward_warning.h:28:2: warning: #warning This file includes at least one deprecated or antiquated header which may be removed without further notice at a future date. Please use a non-deprecated interface with equivalent functionality instead. For a listing of replacement headers and interfaces, consult the file backward_warning.h. To disable this warning use -Wno-deprecated.
    ../src/component/XMLFileCatalog.cpp:25:23: error: uuid/uuid.h: No such file or directory
    ../src/component/XMLFileCatalog.cpp: In function ‘std::string Gaudi::createGuidAsString()’:
    ../src/component/XMLFileCatalog.cpp:176: error: ‘uuid_t’ was not declared in this scope



geant4 : fixed by copying geant4 datafiles to tarFiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://geant4.web.cern.ch/geant4/support/source_archive.shtml

::

    + wget -nv http://cms01.phys.ntu.edu.tw/tarFiles/G4EMLOW.6.2.tar.gz
    http://cms01.phys.ntu.edu.tw/tarFiles/G4EMLOW.6.2.tar.gz:
    2014-07-11 17:43:19 ERROR 404: Not Found.
    + '[' 8 '!=' 0 ']'
    + echo 'Failed to download http://cms01.phys.ntu.edu.tw/tarFiles/G4EMLOW.6.2.tar.gz with command "wget -nv"'
    Failed to download http://cms01.phys.ntu.edu.tw/tarFiles/G4EMLOW.6.2.tar.gz with command "wget -nv"
    + exit 1
    CMT> Error: execution_failed : pkg_install

::

    [blyth@cms01 trunk]$ cp external/geant4/data/*.tar.gz external/tarFiles/



openmotif : fixed needs xorg-x11-xbitmaps 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`18List.c:23:28: error: X11/bitmaps/gray: No such file or directory`

::

    gcc -DHAVE_CONFIG_H -I. -I. -I../../include -I. -I.. -I./.. -DXMBINDDIR_FALLBACK=\"/home/blyth/local/env/dyb/NuWa-trunk/../external/OpenMotif/2.3.0/x86_64-slc6-gcc44-dbg/lib/X11/bindings\" -DINCDIR=\"/home/blyth/local/env/dyb/NuWa-trunk/../external/OpenMotif/2.3.0/x86_64-slc6-gcc44-dbg/include/X11\" -DLIBDIR=\"/home/blyth/local/env/dyb/NuWa-trunk/../external/OpenMotif/2.3.0/x86_64-slc6-gcc44-dbg/lib/X11\" -g -O2 -Wall -g -fno-strict-aliasing -Wno-unused -Wno-comment -fno-tree-ter -I/usr/include/freetype2 -MT I18List.lo -MD -MP -MF .deps/I18List.Tpo -c I18List.c  -fPIC -DPIC -o .libs/I18List.o
    I18List.c:23:28: error: X11/bitmaps/gray: No such file or directory
    I18List.c: In function 'CreateGCs':
    I18List.c:2074: error: 'gray_bits' undeclared (first use in this function)
    I18List.c:2074: error: (Each undeclared identifier is reported only once
    I18List.c:2074: error: for each function it appears in.)
    I18List.c:2075: error: 'gray_width' undeclared (first use in this function)
    I18List.c:2075: error: 'gray_height' undeclared (first use in this function)
    make[3]: *** [I18List.lo] Error 1
    make[3]: Leaving directory `/home/blyth/local/env/dyb/external/build/LCG/openmotif-2.3.0/lib/Xm'


ROOT fails to find /usr/lib64/libXpm.so.4, fixed needs libXpm-devel.x86_64
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     88888 Execute action pkg_get => python /home/blyth/local/env/dyb/NuWa-trunk/lcgcmt/LCG_Builders/LCG_BuildPolicy/scripts/pkg_get.py
     88889 /home/blyth/local/env/dyb/NuWa-trunk/lcgcmt/LCG_Builders/LCG_BuildPolicy/scripts/pkg_get.py : INFO: Downloading
           http://cms01.phys.ntu.edu.tw/tarFiles/root_v5.26.00e.source.tar.gz        to
           /home/blyth/local/env/dyb/NuWa-trunk/../external/tarFiles/root_v5.26.00e.source.tar.gz
     ...
     Checking for X11/extensions/shape.h ... /usr/include
     Checking for libXpm ... no
     configure: libXpm MUST be installed


::

    [blyth@ntugrid5 dyb]$ locate libXpm.so
    /usr/lib64/libXpm.so.4
    /usr/lib64/libXpm.so.4.11.0

::

    [blyth@ntugrid5 root]$ rpm -ql libXpm
    /usr/lib64/libXpm.so.4
    /usr/lib64/libXpm.so.4.11.0
    /usr/share/doc/libXpm-3.5.10
    /usr/share/doc/libXpm-3.5.10/AUTHORS
    /usr/share/doc/libXpm-3.5.10/COPYING
    /usr/share/doc/libXpm-3.5.10/ChangeLog

The problem is lack of the -devel package which has libXpm.so symbolic link
to libXpm.so.4, and maybe needed headers.

* http://www.openmamba.org/distribution/distromatic.html?tag=devel&pkg=libXpm-devel.x86_64











EOU
}
dybinst-dir(){ echo $(local-base)/env/dyb ; }
dybinst-cd(){  cd $(dybinst-dir); }
dybinst-mate(){ mate $(dybinst-dir) ; }
dybinst-url(){     echo http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst ; }
dybinst-get(){
   local dir=$(dybinst-dir) &&  mkdir -p $dir && cd $dir
   [ ! -f dybinst ] && svn export $(dybinst-url)    
}
dybinst-all(){
    dybinst-cd
    ./dybinst -c -u trunk all  
    #screen ./dybinst -c -u trunk all  
}

dybinst-export(){
    export DYB=$(dybinst-dir)
    alias d="cd $DYB"

}

dybinst-check-cms01(){
    curl http://cms01.phys.ntu.edu.tw/tarFiles/
}
dybinst-check-bnl(){
    curl http://dayabay.bnl.gov/software/offline/tarFiles/
}
dybinst-check-ihep(){
    curl http://dayabay.ihep.ac.cn/svn/dybsvn/  --user dayabay:$(cat ~/.dybpass)
}


