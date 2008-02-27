

dybi-get(){
   ## get the branch from the operating directory 
   mkdir -p $DYB
   cd $DYB
   local branch=$(basename $PWD)
   branch=${branch%%_dbg}
   if [ "X$branch" == "Xtrunk" ]; then 
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst
   else
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/branches/inst-NuWa-$branch/dybinst/dybinst
   fi
   echo === dybi-get branch $branch url $url == see https://wiki.bnl.gov/dayabay/index.php?title=Offline_Release_rozz-0.0.4 ==
   svn export $url
}

dybi-check(){
  cd $DYB
  local version=$(basename $PWD)
  version=${version%%_dbg}
  if [ "$version" == "$DYB_VERSION" ]; then
     echo === dybi-check consistent versions $version ==
  else
     echo === dybi-check INCONSITENT VERSIONS ... DYB_VERSION $DYB_VERSION version $version DYB $DYW ===
  fi
}

dybi-linklog(){
  cd $DYB
  rm -f dybinst.log
  local log=$(ls -tr dybinst-*.log|tail -1)
  local cmd="ln -s $log dybinst.log"
  echo === dybi-linklog $cmd ===
  eval $cmd 
}

dybi-log(){
  cd $DYB
  dybi-linklog
  tail -f dybinst.log
}  

dybi-nohup(){
    echo === dyb-install-nohup has a known issue in failing to completely build Geant4, but has advantage of nohup.out summary file ===
    cd $DYB
    rm -f nohup.out
    nohup bash -lc "dyb-ins-install $*"
}

dybi-smry(){
  cd $DYB
  tail -f nohup.out
}  

dybi-install-screen(){
   echo === dybi-install-screen completes the install, but no summary log ... yet === 
   cd $DYB
   screen bash -lc "dybi-install $*"
}


dybi-dbglink(){

   local arch=uncharacterized_linux
   local rootv=5.18.00
   cd $DYB/external/root/$rootv &&  test -d $arch && ln -s $arch ${arch}_dbg || echo dybi-dbglink FAILED 

}


dybi-override(){
   local iwd=$PWD
   cd $DYB
   local override=".dybinstrc"
   if [ "$DYB_OPTION" == "_dbg" ]; then
       echo === dybi-override creating override file $override in folder DYB $DYB
       cat << EOO > $override
# override file created by dyb-override       
gaudi_extra=debug
lhcb_extra=debug
dybgaudi_extra=debug
export ROOTBUILD="debug"
EOO
        cat $override
   else
      echo === dybi-override removing override file $override in folder DYB $DYB
      rm -f $override
   fi
   cd $iwd
}


dybi-install(){
  cd $DYB
  dybi-override
  ./dybinst $DYB_VERSION ${*:-all}
}


  


