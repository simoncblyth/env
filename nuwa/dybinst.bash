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



ROOT fails to find /usr/lib64/libXpm.so.4
----------------------------------------------

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


Maybe needs::

    [blyth@ntugrid5 lcgcmt]$ svn diff LCG_Builders/ROOT/cmt/requirements
    Index: LCG_Builders/ROOT/cmt/requirements
    ===================================================================
    --- LCG_Builders/ROOT/cmt/requirements  (revision 23105)
    +++ LCG_Builders/ROOT/cmt/requirements  (working copy)
    @@ -112,6 +112,11 @@
     
     set LCG_ROOT_CONFIG_OPTIONS ""
     
    +# to allow to locate libXpm.so at eg /usr/lib64/libXpm.so.4
    +set_append LCG_ROOT_CONFIG_OPTIONS "" \
    +   target-x86_64&target-slc6 "--with-xpm-libdir=/usr/lib64 "
    +
    +
     # Xrootd is autodetecting x86_64 on Mac OS X 10.5 - just disable it for now
     set_append LCG_ROOT_CONFIG_OPTIONS "" \
         target-darwin&target-i386 "--disable-xrootd " \


From looking at ROOT configure script this should not be needed.
The problem is lack of the -devel package which has libXpm.so symbolic link
to libXpm.so.4, and maybe needed headers.

::

    [blyth@ntugrid5 root]$ yum info libXpm-devel.x86_64
    Loaded plugins: changelog, kernel-module, priorities, protectbase, refresh-packagekit, security, tsflags, versionlock
    168 packages excluded due to repository priority protections
    126 packages excluded due to repository protections
    Available Packages
    Name        : libXpm-devel
    Arch        : x86_64
    Version     : 3.5.10
    Release     : 2.el6
    Size        : 33 k
    Repo        : slc6-os
    Summary     : X.Org X11 libXpm development package
    URL         : http://www.x.org
    License     : MIT
    Description : X.Org X11 libXpm development package

    [blyth@ntugrid5 root]$ 

    [blyth@ntugrid5 root]$ rpm -ql libXpm-devel.x86_64
    package libXpm-devel.x86_64 is not installed

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


