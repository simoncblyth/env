# === func-gen- : windows/ome/ome fgp windows/ome/ome.bash fgn ome fgh windows/ome
ome-src(){      echo windows/ome/ome.bash ; }
ome-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ome-src)} ; }
ome-vi(){       vi $(ome-source) ; }
ome-env(){      elocal- ; }
ome-usage(){ cat << \EOU

OME 
====

* http://www.openmicroscopy.org/site/support/bio-formats5.1/developers/cpp/overview.html
* https://github.com/ome/ome-cmake-superbuild

This project contains a patch that provides VS 2015 support to xerces-c (see xercesc-) 
and also features a superbuild that pulls in and builds pre-requisite packages
in cross-platform manner.

A good example of involved CMake mechanics.

packages/xerces/superbuild.cmake::

  6 ExternalProject_Add(${EP_PROJECT}
  7   ${OME_EP_COMMON_ARGS}
  8   URL "http://www.apache.org/dist/xerces/c/3/sources/xerces-c-3.1.3.tar.xz"
  9   URL_HASH "SHA512=9931fbf2c91ba2dcb36e5909486c9fc7532420d6f692b1bb24fc93abf3cc67fbd3c9e2ffd443645c93013634000e0bca3ac2ba7ff298d4f5324db9d4d1340600"
 10   SOURCE_DIR "${EP_SOURCE_DIR}"
 11   BINARY_DIR "${EP_BINARY_DIR}"
 12   INSTALL_DIR ""
 13   PATCH_COMMAND


packages/xerces/build.cmake::

      6 if(WIN32)
      7 
      8   message(STATUS "Building xerces (Windows)")
      9 
     10   execute_process(COMMAND msbuild "projects\\Win32\\${XERCES_SOLUTION}\\xerces-all\\xerces-all.sln"
     11                           "/p:Configuration=${XERCES_CONFIG}"
     12                           "/p:Platform=${XERCES_PLATFORM}"
     13                           "/p:useenv=true" "/v:d"
     14                   WORKING_DIRECTORY ${SOURCE_DIR}
     15                   RESULT_VARIABLE build_result)



EOU
}
ome-dir(){ echo $(local-base)/env/windows/ome/ome-cmake-superbuild ; }
ome-fold(){ echo $(dirname $(ome-dir)) ; }

ome-cd(){  cd $(ome-dir); }
ome-get(){
   local dir=$(dirname $(ome-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(basename $dir)
   [ ! -d "$nam" ] && git clone https://github.com/ome/ome-cmake-superbuild
}

ome-xercesc-dir(){      echo $(ome-fold)/$(ome-xercesc-stem) ; }
ome-xercesc-url(){      echo "http://www.apache.org/dist/xerces/c/3/sources/xerces-c-3.1.3.tar.xz" ; }
ome-xercesc-filename(){ echo $(basename $(ome-xercesc-url)) ; }
ome-xercesc-stem(){     local filename=$(ome-xercesc-filename) ; echo ${filename/.tar.xz} ; }
ome-xercesc-patch(){    echo $(ome-dir)/packages/xerces/patches/win-vc14.diff ; }

ome-xercesc-fcd(){      cd $(ome-fold) ; }
ome-xercesc-cd(){      cd $(ome-xercesc-dir) ; }

ome-xercesc-wipe(){
   ( cd $(ome-fold) && rm -rf $(ome-xercesc-stem) )
}

ome-xercesc-get(){
   local dir=$(dirname $(ome-xercesc-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(ome-xercesc-url)
   local xznam=$(basename $url)
   local tarnam=${xznam/.xz}
   local dirnam=${tarnam/.tar}


   echo xznam $xznam tarnam $tarnam dirnam $dirnam

   [ ! -f "$tarnam"  ] && curl -L -O $url 


   if [ -f "$xznam" -a ! -f "$tarnam" ]; then
   
       [ $(which xz) == "" ] && echo $msg no xz : pop over to msys2 to decompress then back here to git bash MGB && return
       xz -d $xznam

   fi

   [ ! -d "$dirnam" ] && tar xvf $tarnam 
   [ ! -d "$dirnam" ] && echo FAILED : tar xvf $tarnam && return 

   if [ -d "$dirnam" -a ! -d "$(ome-xercesc-dir)/projects/Win32/VC14" ]; then
        cd $dirnam 
        patch -p1 < $(ome-xercesc-patch) 
   fi  
   cd $dir
}


ome-xercesc-sln(){
   echo $(ome-xercesc-dir)/projects/Win32/VC14/xerces-all/xerces-all.sln
}

