

dyb--path(){
   echo ${1:-$CMTPATH} | perl -lne 'printf "%s\n",$_ for(split(/:/))'
}

dyb-cmtpath(){ echo === dyb-cmtpath ===  && dyb--path $CMTPATH ; }
dyb-path(){    echo === dyb-path ===     && dyb--path $PATH ; }
dyb-llp(){     echo === dyb-llp ===      && dyb--path $LD_LIBRARY_PATH ; }
dyb-dlp(){     echo === dyb-dlp ===      && dyb--path $DYLD_LIBRARY_PATH ; }


dyb-cmd(){
  
   ##
   ## usage example
   ##    cmd(){ pwd ; ls -alst ; }
   ##    dyb-cmd cmd $PATH
   ##    dyb-cmd cmd $LD_LIBRARY_PATH
   ##
  
   local cmd=${1:-ls}
   local dir=${2:-$PATH}
   local pwd=$PWD
   
   for d in $(dyb--path $dir | grep $DYB)
   do
      if [ -d "$d" ]; then
         cd $d 
         $cmd 
      fi
   done
   
   cd $pwd
}



dyb-info(){
  echo === dyb-info : $* ===
  echo SITEROOT $SITEROOT
  echo CMTPROJECTPATH $CMTPROJECTPATH
  echo CMTEXTRATAGS $CMTEXTRATAGS
  
  echo === which cmt $(which cmt) ===
  echo === which python $(which python) ===
  echo === which root $(which root) ===
  
  dyb-cmtpath
  dyb-path
  dyb-llp
}


dyb-update(){

  cd $DYB
  svn up installation/$DYB_VERSION/dybinst
  svn up $DYB_RELEASE

}




dyb-common(){

  ## 
  ## avoid interactive function/ script issue 
  ##    dirname $0  
  ##          -  gives "." when invoked from interactive bash script
  ##          -  gives absolute path to the directory containing the script, when scripted
  ##

  local instdir=$DYB/installation/$DYB_VERSION/dybinst/scripts
  export BASH_SOURCE=$instdir/virtual-dummy-script-for-interactive-usage-poiposes
  
  ## the dummy is removed by dirname 
  
  source $instdir/dybinst-common.sh
  relver=$DYB_VERSION

}

dyb-make-setup(){  
  
  echo === dyb-make-setup : regenerate the setup directory and scripts in release folder   
  dyb-common
  local config_file=$(main_setup_file $relver sh)
  if [ ! -f $config_file ] ; then
     echo === dyb-setup : creating config_file $config_file 
     make_setup $relver
  else
     echo === dyb-setup : config_file $config_file exists already 
  fi  
}

dyb-unmake-setup(){

  echo === dyb-unmake-setup : attempting  to reset CMT for the project to ground zero 
  cd $DYB
  rm -rf $DYB_RELEASE/setup $DYB_RELEASE/setup.{sh,csh}
  
  unset SITEROOT
  unset CMTPROJECTPATH
  unset CMTEXTRATAGS
  unset CMTPATH         ## suspect this is the critical one  
  
  ##
  ## hmm maybe should cleanup PATH ... or can CMT be pursuaded to do that ?
  ##
}

dyb-setup(){
   
   ## this sets up SITEROOT, CMTPROJECTPATH and CMTEXTRATAGS ... and does generic CMT setup   
   
   local pwd=$PWD
   cd $DYB/$DYB_RELEASE
   . setup.sh 
   cd $pwd
}



dyb-proj(){

   ##
   ## NB cmt gotcha avoided 
   ##   ... have to cd to the directory and then source the setup
   ##  sourcing remotely is not the same DONT YOU JUST LOVE CMT 
   ##

   dyb-unmake-setup
   dyb-make-setup
   dyb-setup

   #dyb-info "prior to project setup " 

   local default="gaudi dybgaudi"
   local proj 
   for proj in ${*:-$default}
   do 
      local rel
      local msg     
      case "$proj" in
          dybgaudi)    rel=dybgaudi/DybRelease         ; msg="this succeeds to setup the path to get the appropriate python "  ;;
             gaudi)    rel=gaudi/GaudiRelease          ; msg="action unknown  "  ;; 
           simualg)    rel=dybgaudi/Simulation/SimuAlg ; msg="untested" ;; 
             hello)    rel=dybgaudi/DybExamples/ExHelloWorld ; msg="untested" ;;
                 *)    rel=NONE ;;
      esac

      local dir=$DYB/$DYB_RELEASE/$rel/cmt
      if [ -d "$dir" -a  -f "$dir/setup.sh" ]; then
         local pwd=$PWD
         echo === dyb-setup-proj $proj : $dir : $msg ==
         cd $dir
         
         ## get rid of the positional parameters, in order to avoid CMT complaint
         set -- 
         cmt config
         . setup.sh
         cd $pwd
      else
         echo === dyb-setup-proj error proj:$proj has no dir $dir or cmt setup file: $dir/setup.sh == 
      fi 
  done
  
  dyb-info "after project setup "
}



dyb-exe-check(){
  local exe=${1:-dyb.exe}
  local xdir=$(dirname $(which $exe))
   if [ "X$xdir" != "X$DYB/$DYB_RELEASE/dybgaudi/InstallArea/$CMTCONFIG/bin" ]; then
      echo === dyb-exe-check the path to $exe is unexpected $xdir 
   else
      echo === dyb-exe-check proceeding xdir $xdir 
   fi
}


dyb-exe(){

   local default=$DYB/$DYB_RELEASE/dybgaudi/Simulation/SimuAlg/share/SimuOptions.txt
   local path=${1:-$default} 
   shift
   local args=$* 
   local exe=dyb.exe
   
   local dir=$(dirname $path)
   local nam=$(basename $path)
   local iwd=$PWD
   
   dyb-proj gaudi dybgaudi
   dyb-exe-check $exe
    
   echo === dyb-exe running $exe from dir $dir on options file $nam ==
   
   cd $dir
   $exe $nam $args
 
   cd $iwd
}


dyb-py(){

   # https://wiki.bnl.gov/dayabay/index.php?title=G4dyb_in_DbyGaudi
   
   local default=$DYB/$DYB_RELEASE/dybgaudi/InstallArea/jobOptions/SimuAlg/RunG4dyb.py
   local path=${1:-$default} 
   shift
   
   local dir=$(dirname $path)
   local nam=$(basename $path)
   local iwd=$PWD

   dyb-proj gaudi dybgaudi 
      
   echo === dyb-py running in dir $dir on py file $nam ==
   
   cd $dir
   ./$nam $*
 
   cd $iwd
   
   #  why did this not land in the path ?
}


