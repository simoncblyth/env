# === func-gen- : juno/offline/offline fgp juno/offline/offline.bash fgn offline fgh juno/offline
offline-src(){      echo juno/offline/offline.bash ; }
offline-source(){   echo ${BASH_SOURCE:-$(env-home)/$(offline-src)} ; }
offline-vi(){       vi $(offline-source) ; }
offline-env(){      elocal- ; }
offline-usage(){ cat << EOU

JUNO Offline 
=============

TODO
----

* move into jnu repo


Observations
-------------

* include svn checkout commandlines in docs

* lots of data files in the repository, 
  perhaps better to split and put data in separate 
  repo 


::

    Checked out revision 2034.
    delta:~ blyth$ du -hs offline
    300M    offline
     

::

    offline-;offline-psm--   # build PMTSim



offline/Detector/Geometry
---------------------------

::

    simon:Geometry blyth$ du -h share/CdGeom.*
     21M    share/CdGeom.gdml
     37M    share/CdGeom.root


Geometry/CdGeom.h
    holder of ROOT TGeo pointers
    CdGeom::readRootGeoFile reads envvar JUNO_GEOMETRY_PATH

Geometry/PmtGeom.h
    center, axis, TGeoPhysicalNode, Identifier

Geometry/RecGeomSvc.h
    RecGeomSvc(SvcBase) holds CdGeom pointer, and file names


Looking for GDML creation 
----------------------------

::

    simon:offline blyth$ find . -name '*.cc' -exec grep -l csv {} \;
    ./Simulation/DetSim/DetSim1/src/LSExpDetectorSolid.cc
    ./Simulation/DetSim/DetSim2/src/LSExpDetectorSolid.cc
    ./Simulation/DetSim/DetSim3/src/LSExpDetectorConstruction.cc
    ./Simulation/DetSim/DetSim3/src/LSExpDetectorMessenger.cc
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc
    ./Simulation/ElecSimV3/UnpackingAlg/elec_sim.cc


* http://juno.ihep.ac.cn/trac/browser/offline/trunk/Simulation/DetSimV2/AnalysisCode/src/GeoAnaMgr.cc


Tao CMake Branches
--------------------

* http://juno.ihep.ac.cn/svn/offline/branches/offline-cmake/


I create a directory for you:
  http://juno.ihep.ac.cn/svn/offline/branches/offline-cmake-simon/
 
But I don’t copy the offline trunk. You can re-organize the whole offline code.

My offline-cmake depends on the SNiPER with cmake support. However it is modified by myself, there is no official release. You can find them in:
  http://juno.ihep.ac.cn/svn/sniper/branches/sniper-cmake/
 


SVN feature branch workflow
-----------------------------

* http://svnbook.red-bean.com/en/1.7/svn.branchmerge.commonpatterns.html
* http://svnbook.red-bean.com/en/1.7/svn.branchmerge.html



Offline CMake Branch (August 2nd, 2016)
------------------------------------------

::

    simon:~ blyth$ offline-;offline-branch 

    Committed revision 2037.



But when the destination url exists already the trunk 
gets put inside, not what is wanted.



EOU
}
offline-dir(){ echo $HOME/$(offline-name) ; }
offline-cd(){  cd $(offline-dir); }

offline-version(){  
   case $USER in
 #     blyth) echo offline-cmake-simon ;; 
          *) echo trunk ;;
   esac 
}

offline-name(){
   local v=${1:-$(offline-version)}
   case $v in
                    trunk) echo offline ;; 
      offline-cmake-simon) echo offline-cmake-simon ;; 
   esac
}

offline-desc(){
   case $v in
                    trunk) echo trunk ;; 
      offline-cmake-simon) echo investigate migration of offline to use CMake configuration ;; 
   esac
}




offline-url(){
   local v=${1:-$(offline-version)}
   case $v in
                  trunk) echo http://juno.ihep.ac.cn/svn/offline/trunk ;; 
      offline-cmake-tao) echo http://juno.ihep.ac.cn/svn/offline/branches/offline-cmake ;; 
    offline-cmake-simon) echo http://juno.ihep.ac.cn/svn/offline/branches/offline-cmake-simon ;; 
   esac
}

offline-get(){
   local dir=$(dirname $(offline-dir)) &&  mkdir -p $dir && cd $dir
   local v=$(offline-version)
   local name=$(offline-name $v) 
   local url=$(offline-url $v) 
   [ ! -d $name ]       && svn co $url $name   
}

offline-find(){
    local cls=${1:-LSExpDetectorConstruction}
    offline-cd
    find . -name $cls.cc
}

offline-rel(){
   case ${1} in 
      DetSim3)  echo Simulation/DetSim/DetSim3  ;; 
      PMTSim)   echo Simulation/DetSimV2/PMTSim ;; 
     SimUtil)   echo Simulation/DetSimV2/SimUtil ;; 
           *)   echo "" ;;
   esac 
}

offline-base(){      echo /usr/local ; }
offline-prefix(){    echo $(offline-base)/jnu/offline ; }
offline-bbase(){     echo $(offline-base)/jnu/offline.build ;  }
offline-bdir(){      echo $(offline-bbase)/$(offline-rel $1) ;  }
offline-sdir(){      echo $(offline-dir)/$(offline-rel $1) ;  }

offline-txt(){       echo $(offline-dir)/$(offline-rel ${1})/CMakeLists.txt ;  }
offline-fnd(){       echo $(offline-dir)/cmake/Modules/Find${1:-PMTSim}.cmake ; }

offline-pkg(){       echo ${PKG:-$1} ; }
offline-tvi(){       vi $(offline-txt  $(offline-pkg $*)) ; } 
offline-fvi(){       vi $(offline-fnd  $(offline-pkg $*)) ; } 
offline-scd(){       cd $(offline-sdir $(offline-pkg $*)) ; }
offline-bcd(){       cd $(offline-bdir $(offline-pkg $*)) ; }


offline-fnd-(){  

   local name=${1:-PMTSim}
cat << EOF

find_library( ${name}_LIBRARIES 
              NAMES ${name}
              PATHS $(offline-prefix)/lib )

if(SUPERBUILD)
    if(NOT ${name}_LIBRARIES)
       set(${name}_LIBRARIES ${name})
    endif()
endif(SUPERBUILD)

# find_package normally yields NOTFOUND
# when no lib is found at configure time : ie when cmake is run
# set the _LIBRARIES to the name of the package 
# which will allow the build to succeed if the target
# is included amongst the add_subdirectory of the super build

set(${name}_INCLUDE_DIRS "\${${name}_SOURCE_DIR}/include")
set(${name}_DEFINITIONS "")

EOF
}

offline-fndgen(){
   local nam=$(basename $PWD)
   local path=$(offline-fnd $nam)
   [ -f "$path" ] && echo $msg path $path exists already && return 

   echo $msg writing $path for pkg $nam 
   mkdir -p $(dirname $path)
   offline-fnd- $nam   > $path 
}



offline-cmake-(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local name=${1}
   local sdir=$(offline-sdir $name)
   local bdir=$(offline-bdir $name)

   cat << EOI

   name : $name
   sdir : $sdir
   bdir : $bdir 


EOI

   mkdir -p $bdir
   cd $bdir

   g4-  
   xercesc-

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(offline-prefix) \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
       $* \
       $sdir

   cd $iwd 
}


offline-make-(){     
   local name=$1
   shift
   local bdir=$(offline-bdir $name)
   offline-build $bdir $*
}


offline-config(){ echo RelWithDebInfo ; }
offline-build(){
   local iwd=$PWD
   local bdir=$1 
   shift  
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return 

   local target=${1:-install} 

   cd $bdir
   cmake --build . --config $(offline-config) --target $target
   cd $iwd 
}


offline-configure()
{
   offline-cmake-
}

offline--()
{
   offline-make-
}





offline-branch(){
   local v=${1:-offline-cmake-simon}
   local desc=$(offline-desc $v)
   svn copy $(offline-url trunk) \
            $(offline-url $v) \
           -m "Copy trunk to branch $v : $desc"

   ## hmm when destination folder exists already this copies the trunk into it
   ##  http://stackoverflow.com/questions/260658/in-svn-how-do-i-copy-just-the-subdirectories-of-a-directory-to-another-director

}

offline-branch-rename(){
   local v=${1:-offline-cmake-simon}
   local url=$(offline-url $v) 

   echo svn rename $url/trunk $url

}

offline-txts-(){  find $(offline-dir) -type f -name CMakeLists.txt ; }
offline-txts(){ vi $(offline-txts-) ; }

offline-fnds-(){  find $(offline-dir) -type f -name Find*.cmake ; }
offline-fnds(){ vi $(offline-fnds-) ; }

offline-rels-(){ cat $(offline-dir)/CMakeLists.txt | perl -n -e 'm,add_subdirectory\((.*)\), && print "$1\n" ' ; }

offline-bdirs-(){
   local bdir
   local rel 
   offline-rels- | while read rel ; do
      bdir=$(offline-bbase)/$rel 
      echo $bdir
   done
}

offline-sdirs-(){
   local sdir
   offline-rels- | while read rel ; do
      sdir=$(offline-dir)/$rel 
      echo $sdir
   done
}


offline---(){
   local bdir
   offline-bdirs- | while read bdir ; do
      printf " %s \n" $bdir  
      offline-build $bdir
   done
}


offline-txtgen-(){
   local name=${1:-GenSim}
   local tmpl=$(offline-dir)/Simulation/DetSimV2/SimUtil/CMakeLists.txt
   cat $tmpl | perl -p -e "s,SimUtil,$name,g" 
}

offline-txtgen(){
   local txt=CMakeLists.txt
   [ -f "$txt" ] && echo $msg txt $txt exists already && return 

   local nam=$(basename $PWD)
   echo $msg writing txt $txt for nam $nam

   offline-txtgen- $nam > $txt
}

offline-gen()
{
   local iwd=$PWD
   local sdir
   offline-sdirs- | while read sdir ; do
      cd $sdir
      offline-txtgen
      offline-fndgen
   done
   cd $iwd
}




