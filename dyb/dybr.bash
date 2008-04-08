

#dybr--path(){ echo === dybr--path : $1 ===  &&  echo $2 | perl -lne 'printf "%s\n",$_ for(split(/:/))' ; }
dybr--path(){ echo === dybr--path : $1 ===  &&  echo $2 | tr ":" "\n" ; }

dybr-cmtpath(){ dybr--path "CMTPATH"           $CMTPATH           ; }
dybr-path(){    dybr--path "PATH"              $PATH              ; }
dybr-llp(){     dybr--path "LD_LIBRARY_PATH"   $LD_LIBRARY_PATH   ; }
dybr-dlp(){     dybr--path "DYLD_LIBRARY_PATH" $DYLD_LIBRARY_PATH ; }
dybr-pypath(){  dybr--path "PYTHONPATH"        $PYTHONPATH        ; }

dybr-cmd(){
  
   ##
   ##  Do something in each directory (thats within $DYB) in the colon delimited string passed 
   ##
   ## usage example
   ##    cmd(){ pwd ; ls -alst ; }
   ##    dybr-cmd cmd $PATH
   ##    dybr-cmd cmd $LD_LIBRARY_PATH
   ##
  
   local cmd=${1:-ls}
   local dir=${2:-$PATH}
   local pwd=$PWD
   
   for d in $(dybr--path $dir | grep $DYB)
   do
      if [ -d "$d" ]; then
         cd $d 
         $cmd 
      fi
   done
   
   cd $pwd
}



dybr-info(){
  
  echo === dybr-info : $* ===
  
  echo SITEROOT $SITEROOT
  echo CMTPROJECTPATH $CMTPROJECTPATH
  echo CMTEXTRATAGS $CMTEXTRATAGS
    
  echo === which cmt $(which cmt) ===
  dybr-cmtpath
   
  echo === which python $(which python) ===
  dybr-pypath
  
  echo === which root $(which root) === ROOTSYS $ROOTSYS
  
  dybr-path

  dybr-llp

  test $(uname) = "Darwin" && dybr-dlp 

}


dybr-common(){

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

  for f in $(env_files $relver) ; do
        source $f
  done

}


dybr-make-setup(){  
  
  echo === dyb-make-setup : regenerate the setup directory and scripts in release folder   
  dybr-common
  local config_file=$(main_setup_file $relver sh)
  if [ ! -f $config_file ] ; then
     echo === dybr-make-setup : creating config_file $config_file 
     make_setup $relver
  else
     echo === dybr-make-setup : config_file $config_file exists already 
  fi  
}

dybr-rm-setup(){

  echo === dybr-rm-setup ===
  cd $DYB
  rm -rf $DYB_RELEASE/setup $DYB_RELEASE/setup.{sh,csh}
  
}


dybr-site-unset(){

   unset SITEROOT
   unset CMTPROJECTPATH

   unset CMTPATH 
  #
  # this avoids a warning on re-running cmt setups (not the 1st run) : 
  #
  #CMT> Project lcgcmt  requested by dybgaudi not found in CMTPROJECTPATH
  #CMT> Project gaudi  requested by dybgaudi not found in CMTPROJECTPATH  
  #CMT> Project lhcb  requested by dybgaudi not found in CMTPROJECTPATH
  #CMT> Project lcgcmt  requested by GAUDI not found in CMTPROJECTPATH
  #CMT> Project gaudi  requested by lhcb not found in CMTPROJECTPATH
  #CMT> Project lcgcmt  requested by lhcb not found in CMTPROJECTPATH
  #
  #

   unset CMTEXTRATAGS
  #
  #  this avoids a warning on re-running dybr-site-setup :
  #CMT> The tag dayabay is not used in any tag expression. Please check spelling
  # 
	
  # 	
  # may be useful to unset or set to ground zero : 
  #     PATH
  #     DYLD_LIBRARY_PATH
  #     LD_LIBRARY_PATH
  #     PYTHONPATH	
  #
  # in extreme debugging cases
  # 
	
	
}


dybr-ss(){ dybr-site-setup $* ; }
dybr-site-setup(){
   
   #
   #  Site setup that needs to be done before most everything else (apart from "dybr-" ) 
   # 
   #     0) clean environment with dybr-site-unset to avoid CMT warnings 
   #     1) generic CMT setup 
   #     2) "cmt config" based on  $DDR/setup/requirements : 
   #           set SITEROOT /disk/d3/dayabay/local/dyb/trunk_dbg/NuWa-trunk
   #           set CMTPROJECTPATH $SITEROOT
   #           set CMTEXTRATAGS dayabay
   #           apply_tag dayabay
   #     3) sources the resulting CMT created $DDR/setup/setup.sh
   #  
   #  NB this may be a good place to add the "debug" to CMTEXTRATAGS
   #
 
   echo === dybr-site-setup ===
   
   dybr-site-unset					   
											   		   
   local pwd=$PWD
   cd $DDR
   
   ## NB this is  concocted by dybinst, not a normal CMT generated from requirements setup.sh
   . setup.sh     
   
   cd $pwd
   
}

dybr-site-info(){

   echo === dybr-site-info ===
   echo SITEROOT $SITEROOT
   echo CMTPROJECTPATH $CMTPROJECTPATH
   echo CMTEXTRATAGS $CMTEXTRATAGS
   which cmt
   cmt show tags

}



dybr-site-initialization(){

   ## only needs doing once ?   as opposed to dybr-site-setup
   ##
   ## NB cmt gotcha avoided 
   ##   ... have to cd to the directory and then source the setup
   ##  sourcing remotely is not the same DONT YOU JUST LOVE CMT 
   ##
   ## 
   ## the point of the below may be to honor the overrides ?
   ## 

   dybr-rm-setup
   dybr-make-setup
   dybr-site-setup

}


dybr-diff(){
   local name="dybr-diff"
   dybr-info > /tmp/$name-before
   $*
   dybr-info > /tmp/$name-after
   diff   /tmp/$name-before /tmp/$name-after

}


dybr-projs(){

  local default_incmt="echo"
  local incmt=${1:-$default_incmt}
  local pwd=$PWD

  echo === dybr-projs [ $incmt ] ===

  dybr-site-setup
  dybr-site-info    
  
  cd $DDR/lcgcmt/LCG_Release   && dybr-cmt $incmt
  cd $DDR/gaudi/GaudiRelease   && dybr-cmt $incmt
  cd $DDR/dybgaudi/DybRelease  && dybr-cmt $incmt

  cd $pwd
}


dybr-cmt(){

   ## use the PWD as the crucial parameter 
   local incmt=$*
   
   local cmtpath=${CMTPATH:-none}
   local cmtextratags=${CMTEXTRATAGS:-none}
   local path=${PATH:-none}
   local dyld_library_path=${DYLD_LIBRARY_PATH:-none}
   local ld_library_path=${LD_LIBRARY_PATH:-none}
   local pythonpath=${PYTHONPATH:-none}
   
   echo === dybr-cmt doing [ $incmt ] and setup in the cmt folder beneath [ $PWD ] ===
   
   if [ -d cmt ]; then
   
		cd cmt
         
		## get rid of the positional parameters, in order to avoid CMT complaint
		set -- 
		$incmt
        if [ ! -f setup.sh ]; then
		   cmt config
		fi
		. setup.sh
		
		cd ..

	else
		echo === dybr-cmt ERROR MUST INVOKE FROM FOLDER WITH A cmt FOLDER  == 
	fi 
	
	local msg="$PWD"
	
	dybr-checkvar "$msg CMTEXTRATAGS"      $cmtextratags      $CMTEXTRATAGS	
	dybr-checkvar "$msg CMTPATH"           $cmtpath           $CMTPATH	
	dybr-checkvar "$msg PATH"              $path              $PATH
	dybr-checkvar "$msg DYLD_LIBRARY_PATH" $dyld_library_path $DYLD_LIBRARY_PATH
	dybr-checkvar "$msg LD_LIBRARY_PATH"   $ld_library_path   $LD_LIBRARY_PATH
	dybr-checkvar "$msg PYTHONPATH"        $pythonpath        $PYTHONPATH
			
							
}

dybr-checkvar(){
   
   local name=$1
   local before=$2
   local after=$3
   
   if [ "$before" == "$after" ]; then
	      echo === $name did not change
   else		   
		  echo === $name CHANGED ==============				
	      dybr--path "before" $before
		  dybr--path "after"  $after
		  echo 
   fi 	
}


dybr-proj-old(){

   dybr-reset

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

      local dir=$DDR/$rel/cmt
      if [ -d "$dir" -a  -f "$dir/setup.sh" ]; then
         local pwd=$PWD
         echo === dybr-proj $proj : $dir : $msg ==
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
  
  dybr-info "after project setup "
}



dybr-xchk(){
  local exe=${1:-dyb.exe}
  local xdir=$(dirname $(which $exe))
   if [ "X$xdir" != "X$DDR/dybgaudi/InstallArea/$CMTCONFIG/bin" ]; then
      echo === dybr-xchk the path to $exe is unexpected $xdir 
   else
      echo === dybr-xchk proceeding xdir $xdir 
   fi
}


dybr-x(){

   local default=$DDR/dybgaudi/Simulation/SimuAlg/share/SimuOptions.txt
   local path=${1:-$default} 
   
   shift
   local args=$* 
   local exe=dyb.exe
   
   local dir=$(dirname $path)
   local nam=$(basename $path)
   local iwd=$PWD
   
   cd $DDR/dybgaudi/Simulation/SimuAlg
   
   dybr-proj gaudi dybgaudi
   dybr-xchk $exe
    
   echo === dybr-x running $exe from dir $dir on options file $nam ==
   
   cd $dir
   $exe $nam $args
 
   cd $iwd
}


dybr-py(){

   # https://wiki.bnl.gov/dayabay/index.php?title=G4dyb_in_DbyGaudi
   
   local default=$DDR/dybgaudi/InstallArea/jobOptions/SimuAlg/RunG4dyb.py
   local path=${1:-$default} 
   shift
   
   local dir=$(dirname $path)
   local nam=$(basename $path)
   local iwd=$PWD

   dybr-proj  
      
   echo === dybr-py running in dir $dir on py file $nam ==
   
   cd $dir
   if [ "X${nam:0:1}" == "Xi" ]; then
      echo === using ipython interactive mode, as script name starts with i ===  
      ipython $nam $*
   else
      ./$nam $*
   fi
 
   cd $iwd
   
   #  why did this not land in the path ?
}


