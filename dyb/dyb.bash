

DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE

dyb-version(){

  # note, sensitivity to preset DYB_VERSION ... overrides the below setting
  export DYB_VERSION_P=trunk
  #export DYB_VERSION_P=0.0.4  
  
 if [ "X$DYB_VERSION" == "X" ]; then
   vname=DYB_VERSION_$NODE_TAG
   eval DYB_VERSION=\$$vname
 else
   echo WARNING honouring a preset DYB_VERSION $DYB_VERSION     
 fi
 
 export DYB_OPTION=""
 #export DYB_OPTION="_dbg"
 
 export DYB_VERSION
 export DYB_FOLDER=$LOCAL_BASE/dyb
 export DYB=$DYB_FOLDER/$DYB_VERSION$DYB_OPTION 
 export DYB_RELEASE=NuWa-$DYB_VERSION

 ## next time distinguish the options (particulary debug on or off status) via the folder name also 

}

dyb-version

dyb(){      [ -r $DYB_HOME/dyb.bash ]           && . $DYB_HOME/dyb.bash ; } 

dyb-get(){
   ## get the branch from the operating directory 
   mkdir -p $DYB
   cd $DYB
   local branch=$(basename $PWD)
   if [ "X$branch" == "Xtrunk" ]; then 
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst
   else
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/branches/inst-NuWa-$branch/dybinst/dybinst
   fi
   echo === dyb-get branch $branch url $url == see https://wiki.bnl.gov/dayabay/index.php?title=Offline_Release_rozz-0.0.4 ==
   svn export $url
}

dyb-check(){
  cd $DYB
  local version=$(basename $PWD)
  if [ "$version" == "$DYB_VERSION" ]; then
     echo === dyb-check consistent versions $version ==
  else
     echo === dyb-check INCONSITENT VERSIONS ... DYB_VERSION $DYB_VERSION version $version DYB $DYW ===
  fi
}

dyb-linklog(){
  cd $DYB
  rm -f dybinst.log
  local log=$(ls -tr dybinst-*.log|tail -1)
  local cmd="ln -s $log dybinst.log"
  echo === dyb-linklog $cmd ===
  eval $cmd 
}

dyb-log(){
  cd $DYB
  dyb-linklog
  tail -f dybinst.log
}  

dyb-install-nohup(){
    echo === dyb-install-nohup has a known issue in failing to completely build Geant4, but has advantage of nohup.out summary file ===
    cd $DYB
    rm -f nohup.out
    nohup bash -lc "dyb-install $*"
}

dyb-smry(){
  cd $DYB
  tail -f nohup.out
}  

dyb-install-screen(){
   echo === dyb-install-screen completes the install, but no summary log ... yet === 
   cd $DYB
   screen bash -lc "dyb-install $*"
}

dyb-install(){
  cd $DYB
  ./dybinst $DYB_VERSION ${*:-all}
}


  





dyb-examples-run(){

  local example=${1:-ExHelloWorld}
  dyb-examples-setup $example
  
  case "$example" in
     ExHelloWorld) invoker=HelloWorldInPython ;;
                *) invoker=NONE ;;
  esac
  
  if [ "X$invoker" == "XNONE" ]; then
     echo === dyb-examples-run  example $example has no corresponding invoker  === 
  else
     which python
     local dir=$PWD 
     cd $DYB
     local cmd="python $DYB_RELEASE/dybgaudi/DybExamples/$example/share/$invoker.py"
     echo $cmd
     eval $cmd
     cd $dir 
  fi  

}

dyb-examples-setup(){

   local example=${1:-ExHelloWorld}
   local dir=$PWD
   
   dyb_proj gaudi dybgaudi hello
        
   which python
   cd $dir
}


dyb-exe(){

   local dir=$PWD
   dyb_proj
   
   which dyb.exe
   ##cd $DYB/$DYB_RELEASE/dybgaudi/InstallArea/$CMTCONFIG/bin
   
   cd $DYB/$DYB_RELEASE/dybgaudi/Simulation/SimuAlg/share/
   dyb.exe SimuOptions.txt
 
   cd $dir
}




dyb-info(){
  echo SITEROOT $SITEROOT
  echo CMTPROJECTPATH $CMTPROJECTPATH
  echo CMTEXTRATAGS $CMTEXTRATAGS
  echo === which cmt $(which cmt) ===
  echo === which python $(which python) ===
  
  dyb-cmtpath
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
  export BASH_SOURCE=$instdir
  
  source $instdir/dybinst-common.sh
  relver=$DYB_VERSION

}

dyb-make-setup(){  
  
  echo === dyb-setup : regenerate the setup directory and scripts in release folder   
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

  echo === dyb-clear : attempting  to reset CMT for the project to ground zero 
  cd $DYB
  rm -rf $DYB_RELEASE/setup $DYB_RELEASE/setup.{sh,csh}
  unset SITEROOT
  unset CMTPROJECTPATH
  unset CMTEXTRATAGS
  unset CMTPATH         ## suspect this is the critical one  
}

dyb-setup(){
   
   ## this sets up SITEROOT, CMTPROJECTPATH and CMTEXTRATAGS ... and does generic CMT setup   
   
   local pwd=$PWD
   cd $DYB/$DYB_RELEASE
   . setup.sh 
   cd $pwd
}

dyb-cmtpath(){
  echo $CMTPATH | perl -lne 'printf "%s\n",$_ for(split(/:/))'
}

dyb-proj(){

   ##
   ## NB cmt gotcha avoided 
   ##   ... have to cd to the directory and then source the setup
   ##  sourcing remotely is not the same DONT YOU JUST LOVE CMT 
   ##
   ## 
   ## hmmm after this are missing all the dependents :  
   ##   echo $CMTPATH
   ##   /disk/d4/dayabay/local/dyb/trunk/NuWa-trunk/dybgaudi:/disk/d4/dayabay/local/dyb/trunk/NuWa-trunk/gaudi
   ##
   ##  next : examine CMTPATH during the build and runtime preparation 
   ##
   ##

   dyb-unmake-setup
   dyb-make-setup
   dyb-setup


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
         cmt config
         . setup.sh
         cd $pwd
      else
         echo === dyb-setup-proj error proj:$proj has no dir $dir or cmt setup file: $dir/setup.sh == 
      fi 
  done
  
   ##
   ## would you avoid the environment dependency headache, by just plucking 
   ## the appropriate python and letting it handle it ?
   ##

}

dyb-sim(){

   # https://wiki.bnl.gov/dayabay/index.php?title=G4dyb_in_DbyGaudi
   
   dyb-proj gaudi dybgaudi 
   which python
   
   cd $DYB/$DYB_RELEASE/dybgaudi/InstallArea/jobOptions/SimuAlg/
   ./RunG4dyb.py

}



dyb-sleep(){
  sleep $* && echo "dyb-sleep completed $* " > /tmp/dyb-sleep
}  
  









