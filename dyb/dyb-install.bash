

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


  


