

dybi-usage(){

cat << EOU

   dybi-info            :   dump envvars used in these functions
   dybi-get             :   export dybinst script into $DYB
   dybi-update          :   update the installation and release 
   dybi-check           :   version check between DYB and DYB_VERSION
   
   dybi-common          :   invoke the installation/scripts/common.sh providing
                                   cmt_macro <pkg> <macro>
                                        exports a macro value into macro
                                   check_pkg  <pkg>     
                                        checks existance of $SITEROOT/lcgcmt/LCG_Interfaces/$pkg/cmt/requirements
   
   
   #dybi-log             :   do the -linklog and follow the installation tail 
   #dybi-linklog         :   link the installation log ... now done automatically ?
   #
   
   dybi-nohup           :   nohuped invokation of dybi-install 
                            has a known issue in failing to completely build Geant4, 
						    but has advantage of nohup.out summary file
   dybi-nohup-tail
   
   dybi-install-screen  :  the install thru screen for disconnection immunity  
   
   dybi-install         :  run dybinst 
   dybi-tail            :  tail -f on the dybinst-recent.log  
   
   dybi-dbglink         :  create the link needed for debug installs ... ? did i automate this ?
   dybi-override        :  create the "_extra" files that switch on debug 

   dybi-osc-zip         :  zip path   
   dybi-osc-patch-path  :  patch path  
   dybi-osc-xclude      :  detritus exclusion 
   dybi-osc-diff        :  diff to stdout 
   dybi-osc-patch-test  :  
   dybi-osc-patch       :  uses dybi-osc-diff to write the patch
   dybi-osc             :  allows building sub packages of OpenScientist 
   
   
   dybi-external        :  dybinst-external build 
                             name must correspond to the script name ... all lowercase
                                  dybi-;dybi-external qmtest 
                
        
   
   
EOU

}



dybi-env(){

   elocal-
   dyb-
}



dybi-external(){

   local iwd=$PWD
   
   dybr-
   dybr-ss
   
   logfile=/dev/null
   $DYB/installation/trunk/dybinst/scripts/dybinst-external $DYB_VERSION $*
   
   cd $iwd

}


dybi-common(){

   local iwd=$PWD
   
   dybr-
   dybr-ss
   
   . $DYB/installation/trunk/dybinst/scripts/common.sh
   cd $iwd

}


dybi-info(){

   echo DYB $DYB
   echo DDR $DDR
   echo DDI $DDI
   echo DYB_VERSION $DYB_VERSION 
   echo DYB__RELEASE $DYB__RELEASE 
   echo DYB_OPTION $DYB_OPTION
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
  svn up $DYB__RELEASE
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

#dybi-linklog(){
#  cd $DYB
#  rm -f dybinst.log
#  local log=$(ls -tr dybinst-*.log|tail -1)
#  local cmd="ln -s $log dybinst.log"
#  echo === dybi-linklog $cmd ===
#  eval $cmd 
#}

dybi-tail(){
  cd $DYB
  #dybi-linklog
  tail -f dybinst-recent.log
}  

dybi-nohup(){
    echo === dyb-install-nohup  ===
    cd $DYB
    rm -f nohup.out
    nohup bash -lc "dybi-install $*"
}

dybi-nohup-tail(){
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


  


