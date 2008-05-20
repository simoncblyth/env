

dybi-env(){

   elocal-

}



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

dybi-update(){

  cd $DYB
  svn up installation/$DYB_VERSION/dybinst
  svn up $DYB_RELEASE
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
   screen bash -lc "dybi- ; dybi-install $*"
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




dybi-osc-zip(){
   local ver=v16r1   
   echo  $DYB/external/OpenScientist/src/osc_source_$ver.zip    
}

dybi-osc-patch-path(){
   local name=$(basename $(dybi-osc-zip))
   local path=$DDI/../patches/$name.patch 
   echo $path
}

dybi-osc-xclude(){
   local xcl=" -x "*.sh" -x "*.csh" -x "*.bat" -x foreign -x sh -x bin_obuild -x .DS_Store "
   echo $xcl 
}

dybi-osc-diff(){
   local def_opt="-r --brief"
   local opt=${1:-$def_opt}
   patch-
   patch-diff $(dybi-osc-zip) OpenScientist "$opt $(dybi-osc-xclude)"
}

dybi-osc-patch-test(){
  local def_opt="-r --brief"
  local opt=${1:-$def_opt}
  patch-
  patch-test $(dybi-osc-zip) OpenScientist $(dybi-osc-patch-path) "$opt $(dybi-osc-xclude)"
}

dybi-osc-patch(){
  local def_opt="-Naur"
  local opt=${1:-$def_opt} 
  local path=$(dybi-osc-patch-path)
  echo === $0/$FUNCNAME : writing patch to $path
  dybi-osc-diff $opt > $path
}




dybi-osc(){

  local xpkg=${1:-CoinGL} 
  local opt="-v"  
  
  ## -l for "link only" not working... get no object files
  
  local pwd=$PWD
  
  local ver=v16r1
  local src=$DYB/external/OpenScientist/src/OpenScientist/$ver
  local vis=$src/osc_vis/$ver/obuild  

  if [ -d "$vis" ]; then
     cd $vis
     source setup.sh
  else
     echo === dybi-osc : ERROR openscientist version in dybi.bash is outdated $vis 	  
  fi
  
  which obuild
  
  obuild -q | grep package | while read pkg colon what ver
  do
    
     if [ "$xpkg" == "$what" ]; then
	
	     local dir=$src/$xpkg/$ver/obuild
		 echo === dybi-osc : $what $pkg $ver : proceed : $dir === 
	     
		 
		 if [ -d "$dir" ]; then
		 
		    cd $dir
			
			echo === dybi-osc : configuring in $dir
			obuild 
			
			echo === dybi-osc : building in $dir : opt $opt :  -l link only -v verbose
			./sh/build $opt 
			
			echo === dybi-osc : building for group Python in $dir
			./sh/build $opt -group Python
			
		 else
		    echo === dybi-osc ERROR no folder $dir
		 fi

     else
		 echo === dybi-osc : $what $pkg $ver : skip  ===	
	 fi
	 
	 
  done
	 	 
		 
		 
}


  


