nuwa-src(){    echo nuwa/nuwa.bash ; }
nuwa-source(){ echo ${BASH_SOURCE:-$(env-home)/$(nuwa-src)} ; }
nuwa-vi(){     vi $(nuwa-source) ; }
nuwa-utils(){
  cat << EOX

       nuwa-genpkgdir 
           generate the nuwa-pkgdir function based on the output of "cmt show packages"

       nuwa-pkgdir <pkg>
           echo absolute path of the named package 
           ... the matching is done case-sensitively first and then with all lowercased 

       nuwa-cd                                      (this is aliased to "ncd" )
          cd to the directory obtained from nuwa-dir, arguments passed as is
          
       nuwa-dir  <pkg>[/aaa/bbb/etc] <sub>          (this is aliased to "nd" )
          echo absolute path according to argument specification...
          if the first argument contains a slash, the first element is
          interpreted as the CMT package ... which is used to lookup the absolute package directory, 
          the subsequent elements and the 2nd argument (if present) specify a subpath 
          within the project
            eg 
              nd gentools cmt
              nd gentools/cmt
              "$(nd rootiotest/tests)" == "/data/env/local/dyb/trunk_dbg/NuWa-trunk/dybgaudi/RootIO/RootIOTest/tests"

          NB this is just doing a case lookup... so it is very fast 

     TODO :
        also parse "cmt show projects" ... allowing to make more sense of the path 
        allowing for example the BUILD_PATH in .bash_profile to be derived from a project name

        factor out the cmt parsing ... 
            as it is in fact not nuwa specific and can be rehomed into cmt- for 
            usage in any cmt ensemble


        reposition the warning about no case function defined .. to when you want to use it
        rather than every new session


Greenfield build took about 5hrs::

    [blyth@belle7 dyb]$ ll
    total 32928
    -rwxrwxr-x  1 blyth blyth    29654 May 10  2013 dybinst
    drwxr-xr-x  9 blyth root      4096 Mar  3 14:54 ..
    drwxrwxr-x  3 blyth blyth     4096 Mar  3 14:55 installation
    lrwxrwxrwx  1 blyth blyth       27 Mar  3 14:55 dybinst-recent.log -> dybinst-20140303-145546.log
    drwxrwxr-x 10 blyth blyth     4096 Mar  3 14:57 NuWa-trunk
    drwxrwxr-x  5 blyth blyth     4096 Mar  3 16:00 .
    drwxrwxr-x 33 blyth blyth     4096 Mar  3 17:21 external
    -rw-rw-r--  1 blyth blyth 33617403 Mar  3 20:04 dybinst-20140303-145546.log
    [blyth@belle7 dyb]$ cd external/





EOX



}


nuwa-env(){ 
   elocal- 
}

nuwa-plat(){ 
  case $NODE_TAG in
     N) echo i686-slc5-gcc41-dbg ;;
     C) echo i686-slc4-gcc34-dbg ;;
  esac
}

nuwa-dyb(){ 
   case ${DYB:0:1} in
     /) echo $DYB ;;                   ## absolute DYB just provide asis
     *) echo ${LOCAL_BASE}/dyb$DYB ;;  ## relative, prefix with LOCAL_BASE/dyb
   esac
}

nuwa-dybgaudi-dir(){ echo $(nuwa-dyb)/NuWa-trunk/dybgaudi ; } 
nuwa-lcgcmt-dir(){ echo $(nuwa-dyb)/NuWa-trunk/lcgcmt ; } 
nuwa-lhcb-dir(){ echo $(nuwa-dyb)/NuWa-trunk/lhcb ; } 
nuwa-g4-bdir(){ echo $(nuwa-dyb)/external/build/LCG/geant4.9.2.p01 ; }
nuwa-g4-bdir-old(){ echo $(nuwa-dyb)/../dyb.old/external/build/LCG/geant4.9.2.p01 ; }
nuwa-g4-idir(){ echo $(nuwa-dyb)/external/geant4/4.9.2.p01/$(nuwa-plat) ; }
nuwa-g4-cmtdir(){ echo $(nuwa-lcgcmt-dir)/LCG_Builders/geant4/cmt ; }
nuwa-g4-pdir(){ echo $(nuwa-lcgcmt-dir)/LCG_Builders/geant4/patches ; }
nuwa-pkg-cmtdir(){ echo $(nuwa-lcgcmt-dir)/LCG_Builders/$1/cmt ; }
nuwa-g4-xdir(){ echo $(nuwa-g4-bdir)/bin/Linux-g++ ; }
nuwa-g4-sdir(){ echo $(nuwa-g4-bdir)/source ; }

nuwa-g4-incdir(){ echo $(nuwa-g4-idir)/include ; }
nuwa-g4-libdir(){ echo $(nuwa-g4-idir)/lib ; }

nuwa-g4-wipe-marker(){
  rm $(nuwa-g4-bdir)/lib/Linux-g++/libG4run.so
  rm $(nuwa-g4-bdir)/lib/Linux-g++/libname.map
  rm $(nuwa-g4-bdir)/include/G4Version.hh 
}


nuwa-clhep-ver(){  echo 2.0.4.2 ; }
nuwa-clhep-bdir(){ echo $(nuwa-dyb)/external/build/LCG/clhep/$(nuwa-clhep-ver) ; }
nuwa-clhep-idir(){ echo $(nuwa-dyb)/external/clhep/$(nuwa-clhep-ver)/$(nuwa-plat) ; } 
nuwa-clhep-lib(){ echo CLHEP-$(nuwa-clhep-ver) ; } 
nuwa-clhep-incdir(){ echo $(nuwa-clhep-idir)/include ; } 
nuwa-clhep-libdir(){ echo $(nuwa-clhep-idir)/lib ; } 


nuwa-xercesc-bdir(){ echo $(nuwa-dyb)/external/build/LCG/xerces-c-src_2_8_0 ; }
nuwa-xercesc-idir(){ echo $(nuwa-dyb)/external/XercesC/2.8.0/$(nuwa-plat) ; }
nuwa-xercesc-incdir(){ echo $(nuwa-xercesc-idir)/include ; }
nuwa-xercesc-libdir(){ echo $(nuwa-xercesc-idir)/lib ; }

nuwa-python-idir(){ echo $(nuwa-dyb)/external/Python/2.7/$(nuwa-plat) ; }
nuwa-python-incdir(){ echo $(nuwa-python-idir)/include/python2.7 ; }
nuwa-python-libdir(){ echo $(nuwa-python-idir)/lib ; }

nuwa-export(){
  export NUWA_G4_INCDIR=$(nuwa-g4-incdir)
  export NUWA_G4_LIBDIR=$(nuwa-g4-libdir)
  export NUWA_CLHEP_LIBDIR=$(nuwa-clhep-libdir)
  export NUWA_CLHEP_INCDIR=$(nuwa-clhep-incdir)
  export NUWA_CLHEP_LIB=$(nuwa-clhep-lib)
  export NUWA_XERCESC_LIBDIR=$(nuwa-xercesc-libdir)
  export NUWA_XERCESC_INCDIR=$(nuwa-xercesc-incdir)
 
}



nuwa-env-deprecated(){ 
   local msg="=== $FUNCNAME :"
   local v=$(nuwa-version $*)
   if [ "$(nuwa-isinstalled $*)" == "NO" ]; then
      [ "$v" == "trunk" ] && echo $msg ABORT trunk is not installed .. infinite recursion avoidance && return 1
      echo $msg WARNING nuwa IS NOT installed based on "$*" OR NUWA_HOME : $NUWA_HOME , attempt fallback to trunk 
      nuwa- trunk
   else
      [ -z "$NUWA_NOFUNC" ] && nuwa-functions $*
      nuwa-exports $*   
      #nuwa-defpkgdir
   fi
}




nuwa--(){     screen bash -lc "nuwa- ; $*" ; } 

nuwa-first(){  echo $1 ; }
nuwa-second(){ echo $2 ; }
nuwa-third(){  echo $3 ; }
nuwa-lower() { echo $1 | tr "[:upper:]" "[:lower:]"  ; } 

nuwa-genpkgdir-(){

    
   local line
   echo "nuwa-pkgdir(){ "
   echo "## caution this function was generated by $BASH_SOURCE::$FUNCNAME on $(date) "
   echo "## based on \"cmt show packages\" performed from  $PWD "
   echo " case \$1 in "

   ## let correct case ... have first try    
   cmt show packages | while read line ; do
      nuwa-parse "$line"  
   done 

   ## then with case lowered 
   cmt show packages | while read line ; do
      nuwa-parse "$line"  lower
   done 

   echo "*) echo .  ;; "
   echo "esac "
   echo "} "
}


nuwa-parse(){
   local line="$1"
   local meth=$2

   local pkg=$(nuwa-first  $line)
   local lkg=$(nuwa-lower $pkg)
   local dir
   case $pkg in 
        VisRelease|DetRelease)  dir=$(nuwa-second  $line) ;; 
                            *)  dir=$(nuwa-third  $line)  ;; 
   esac

   ## keep absolute to work with CMTPROJECTPATH tethered layouts
   ## dir=${dir/$NUWA_HOME/}

   case $meth in 
     lower) [ "$lkg" != "$pkg" ] && echo   "$lkg) echo $dir/$pkg ;; "  ;;
         *) echo   "$pkg) echo $dir/$pkg ;; "  ;;
   esac  
}


nuwa-tmpdir(){  echo /tmp/env/${FUNCNAME/-*/}; }
nuwa-genpath(){ echo $(nuwa-tmpdir)/$FUNCNAME.gen.bash ; }
nuwa-genpkgdir(){
  local iwd=$PWD
  dyb__ dybgaudi/DybRelease/cmt
  local msg="=== $FUNCNAME :"
  local tmp=$(nuwa-tmpdir) && mkdir -p $tmp
  local gen=$(nuwa-genpath)
  [ ! -f "$gen" ] && echo $msg generating pkg to dir case statement based function nuwa-pkgdir && nuwa-genpkgdir- > $gen
  [   -f "$gen" ] && echo $msg sourcing $gen providing nuwa-pkgdir function ... delete this to regenerate
  . $gen
  cd $iwd
}

nuwa-defpkgdir(){
  local gen=$(nuwa-genpath)
  [ ! -f "$gen" ] &&  echo $msg $(nuwa-source) : ERROR you need to generate \"$gen\" which should contain the func nuwa-pkgdir with nuwa-genpkgdir && return 1
  . "$gen" 
}


nuwa-dir(){
  local arg=$1

  [ -z "$arg" ] && echo $(nuwa-base) && return 0
  local pkg
  local sub


  ## arg contains a slash ... 
  if [ "${arg/\//}" != "$arg" ]; then
     pkg=${arg/\/*/}
     sub=${arg:$((${#pkg}+1))} 
     [ -n "$2" ] && sub="$sub/$2" 
 else
     pkg=${1:-DybRelease}
     sub=$2
  fi
  [ -z "$sub" ] &&  echo $(nuwa-pkgdir $pkg) || echo $(nuwa-pkgdir $pkg)/$sub
}
alias nd="nuwa-dir"


nuwa-cd(){
  cd $(nuwa-dir $*)
}
alias ncd="nuwa-cd"



nuwa-usage(){
  cat << EOU


   NB see also "nuwa-utils" for info on the tools "nd" "ncd" for jumping
      around from project to project 

   Versioning and placement control based on a single variable : NUWA_HOME

           NUWA_HOME              :  $NUWA_HOME
           nuwa-home-default      :  $(nuwa-home-default)

   
   If defined the NUWA_HOME must follow a standard layout for its last 2 levels, eg 
        /whatever/prefix/is/desired/1.0.0-rc02/NuWa-1.0.0-rc02
   if not defined a default is used


   Usage :
            nuwa-functions <v>
                source the dyb__.sh and slave.bash from for version v, defaulting to trunk 
                (equivalent to the former dyb-hookup )

            nuwa-info <v>
                dump the distro parameters for version v, defaulting to trunk 

            nuwa-exports <v>
                export convenience variables
                NB NUWA_HOME is never exported, that is regarded as an input only 

       
   Example of usage in .bash_profile :
   
        ## defines the nuwa- precursor , so must come before that 
        env-     
        
        ## comment the below two lines to use the default of "trunk"
        export NUWA_VERSION=1.0.0-rc02 ... used by dyb__version function for dybinst option setting 
        export NUWA_HOME=whatever/$NUWA_VERSION/NuWa-$NUWA_VERSION   ## NB prior to invoking "nuwa-" precursor 
        
        ## defines the functions and exports  
        nuwa-       
             
        ## setup a default location for operations : building/testing      
        export BUILD_PATH=dybgaudi/${nuwa_version:-trunk}/RootIO/RootIOTest
             
             
             
    To temporarily jump into another release :
        
        nuwa- 1.0.0-rc01          ( OR  trunk ) 
        nuwa-info    
        
     ## NB the dynamics will still be based using the version from NUWA_HOME, but the 
        exports and function paths should now show they hail from the chosen release
        



    dybinst release 1.0.1 testing ... where the trunk externals are re-used
    
       1) Set NUWA_HOME in .bash_profile corresponding to release 1.0.1
       2) from that environment  
            NUWA_DYBINST_OPTIONS="-e $(nuwa-external trunk)" nuwa-dybinst     

    to do this thru screen ... export NUWA_DYBINST_OPTIONS in the .bash_profile 





    THOUGHTS :
         nuwa- trunk
    
       tiz very useful to be able to use latest improvements to functions from trunk 
       while acting on behind the times releases ... so the functions should be 
       setup with this in mind, namely the envvar controlled version selection
       takes priority over the BASH_SOURCE determined one 
      



EOU

   nuwa-info


}

nuwa-dump(){

   nuwa-info
   nuwa-info trunk
   nuwa-info 1.0.0-rc01
   nuwa-info 1.0.0-rc02

}



nuwa-info(){
  local v=$1
  cat << EOI
  
   Dynamically derived quantities for version provided $v 
   If no version argument is given determine the quantities based
   in the value of  NUWA_HOME : $NUWA_HOME 
   
   NB unfortunately dyb__* functions have some dependency on NUWA_VERSION too
       NUWA_VERSION : $NUWA_VERSION  
   
   
   OR a default if that is not defined 
  
          nuwa-home $v    :  $(nuwa-home $v)
          nuwa-base $v    :  $(nuwa-base $v)
          nuwa-release $v :  $(nuwa-release $v)
          nuwa-version $v :  $(nuwa-version $v)
          nuwa-base $v    :  $(nuwa-base $v) 
          nuwa-scripts $v :  $(nuwa-scripts $v)  
          nuwa-dyb__ $v   :  $(nuwa-dyb__ $v)   
          nuwa-slave $v   :  $(nuwa-slave $v)  
  
      
          nuwa-ddr $v     :  $(nuwa-ddr $v)       
          nuwa-ddi $v     :  $(nuwa-ddi $v)        
          nuwa-ddt $v     :  $(nuwa-ddt $v)       
  
    Exported into environment :
        
            DYB : $DYB
            DDR : $DDR
            DDI : $DDI
            DDT : $DDT
            


    For dybinst/release testing ..  (NB quoting caution, as echo swallows -e )
            
          NUWA_DYBINST_OPTIONS    : $NUWA_DYBINST_OPTIONS  
          nuwa-dybinst-options $v : $(nuwa-dybinst-options $v)    
          nuwa-dybinst-cmd $v     : $(nuwa-dybinst-cmd $v)
 
    For screen protection :
         nuwa-- nuwa-dybinst projects dybgaudi

    for full build : 
         nuwa-- nuwa-dybinst all 
         
    Follow what happening with tail on the log and : 
         pstree -Gapu $USER         

   
            
    Source paths reported by the functions hooked up into the environment :
    
         dyb__source : $(dyb__source)
         slave-path  : $(slave-path) 
            
    Checking to see if nuwa for version \"$v\" is installed already based on existance of the dyb__.sh      
  
         nuwa-isinstalled $v : $(nuwa-isinstalled $v)
  
EOI
}



nuwa-home-default(){  echo $LOCAL_BASE/dyb/trunk/NuWa-trunk ; }

nuwa-home-construct(){
   local v=$1
   case $v in
      trunk) nuwa-home-default ;;
          *) echo $LOCAL_BASE/dyb/releases/$1/NuWa-$1 ;;
   esac       
}

nuwa-home(){          
  if [ "$1" == "" ]; then
     echo ${NUWA_HOME:-$(nuwa-home-default)}
  else 
     nuwa-home-construct $1 
  fi
}

nuwa-release(){       echo $(basename $(nuwa-home $*)); }
nuwa-version(){       local rel=$(nuwa-release $*) ; echo ${rel/NuWa-/} ; }
nuwa-scripts(){       echo installation/$(nuwa-version $*)/dybtest/scripts ; }

nuwa-base(){          echo $(dirname $(nuwa-home $*)) ; } 
nuwa-dyb-deprecated(){           echo $(nuwa-base $*) ; } 
nuwa-external(){      echo $(nuwa-base $*)/external ; } 
nuwa-ddi(){           echo $(nuwa-base $*)/installation/$(nuwa-version $*)/dybinst/scripts ; }
nuwa-ddt(){           echo $(nuwa-base $*)/installation/$(nuwa-version $*)/dybtest ; }
nuwa-dyb__(){         echo $(nuwa-base $*)/$(nuwa-scripts $*)/dyb__.sh ; }
nuwa-slave(){         echo $(nuwa-base $*)/$(nuwa-scripts $*)/slave.bash ; }
nuwa-daily(){         echo $(nuwa-base $*)/$(nuwa-scripts $*)/daily.bash ; }

nuwa-ddr(){           echo $(nuwa-home $*) ; } 
nuwa-ddp(){           echo $(nuwa-home $*)/dybgaudi/DybPython/python/DybPython ; }
nuwa-ddx(){           echo $(nuwa-home $*)/tutorial/Simulation/SimHistsExample/tests ; }
nuwa-log(){           echo $(nuwa-base $*)/dybinst-recent.log ; }
nuwa-tail(){          tail -f $(nuwa-log $*) ; }

nuwa-exports(){
   export DYB=$(nuwa-dyb $*)
   export DDT=$(nuwa-ddt $*)
   export DDP=$(nuwa-ddp $*)
   export DDI=$(nuwa-ddi $*)
   export DDR=$(nuwa-ddr $*)
   export DDX=$(nuwa-ddx $*)
}

nuwa-functions(){     
   
    local msg="=== $FUNCNAME : "
    local dyb__=$(nuwa-dyb__ $*)
    local slave=$(nuwa-slave $*)
    local daily=$(nuwa-daily $*)
    
    if [ -f $dyb__ ]; then
        set --
        . $dyb__
        [ -z $BASH_SOURCE ] && eval "function dyb__source(){  echo $dyb__ ; }"      ## workaround for older bash  
        dyb__default(){ echo dybgaudi/Simulation/GenTools ; } 
    else
        echo $msg no dyb__ $dyb__
    fi 
     
    if [ -f $slave ]; then
       . $slave
    else
       echo $msg no slave $slave 
    fi

    if [ -f $daily ]; then
       . $daily
    else
       echo $msg no daily $daily 
    fi


}

nuwa-isinstalled(){
   nuwa-isinstalled- $* && echo YES || echo NO
}

nuwa-isinstalled-(){
    local dyb__=$(nuwa-dyb__ $*)
    [ -f "$dyb__" ] && return 0 || return 1
}



nuwa-all(){  nuwa-- nuwa-dybinst all ; }  ## with screen protection 
nuwa-dybinst-url(){     echo http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst ; }
nuwa-dybinst-cmd(){     echo ./dybinst "$(nuwa-dybinst-options)" $(nuwa-version)  ; }
nuwa-dybinst-options(){ echo "${NUWA_DYBINST_OPTIONS:-""}" ; }
nuwa-dybinst(){

    local args=$*
    local msg="=== $FUNCNAME :"
    echo $msg args $args

    local base=$(nuwa-base)
    [ ! -d "$base" ] && echo $msg creating base directory $base 
    
    mkdir -p "$base"
    cd "$base"
    
    local url=$(nuwa-dybinst-url)
    [ ! -f dybinst ] && echo $msg exporting $url && svn export $url
    
    local cmd="$(nuwa-dybinst-cmd) $args"
    local ans
    read -p "$msg from $PWD proceed with : [ $cmd ]  enter YES to proceed : " ans     
    [ "$ans" != "YES" ] && echo $msg skipped ... && return 1
    
    echo $msg proceeding ...
    eval $cmd
    

}

nuwa-fromscratch(){
   cd $DYB
   nuwa-dybinst-export
   [ "$(which python)" != "/usr/bin/python" ] && echo use system python for clean rebuilds && return 1
   screen ./dybinst trunk all
}

nuwa-dybinst-export(){
   svn export $(nuwa-dybinst-url)
}
nuwa-xfromscratch(){
   cd $DYBX
   [ "$(which python)" != "/usr/bin/python" ] && echo use system python for clean rebuilds && return 1
   screen ./dybinst -X geant4_with_dae trunk all
}




