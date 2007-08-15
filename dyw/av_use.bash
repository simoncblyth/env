#
#
#   av-use-sub         submit the autovalidation jobs to condor ... with xml logging
#   autovalidation     a wrapper for av-use-run in order to have primary element name in the xml
#   av-use-run         run the autovalidation perl script 
#   av-use-today
#   av-use-cf          not implemented
#
#
#  usage :
#
#     submit autovalidation to the condor batch system on grid1 with 
#     (NB this does not work as dayabaysoft user)
#
#       G1> av-use-sub
#
#   some hrs later look at the dated results at 
#        http://grid1.phys.ntu.edu.tw:8080/autovalidation/
#
#   see $DYW_BASE/av.bash for detailed description of what the autovalidation machinery is doing 
#
#
#
#   TODO:  extract the numbers from the root file in a script... for automated
#          event reproducibility
#
# Hi jenarron,
#
# You should add two commands in the macro files of the Auto Validation job.
# (eminus_1MeV.mac, gamma_1MeV.mac, muon_100GeV.mac and neutron_1keV.mac)
# /dyw/run/runNumber number_1
# /dyw/run/hostID    number_2
# number_1 and number_2 can be found in the output root file.(runID and hostID)
# As for where to insert the two commands, please refer to 
# G4dyb/example/test_reproducibility.mac. 
#
# Best regards,
# Liang Zhan
# 2007-05-25
#
#
[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/av_use.bash



av-use-build(){

  local iwd=$PWD
  local branch=${1:-$DYW_VERSION}
  
  echo ==== av-use-build building G4dybApp.exe from scratch with the latest from branch $dyw/branches/$branch
  cd $DYW_FOLDER
  
  if [ -d "$branch" ]; then
     cd $branch
     svn up 
  else
     svn co $dyw/branches/$branch
  fi

  cd $DYW_FOLDER/$branch/G4dyb/cmt
  
  local flags="CMTEXTRATAGS=debug TMP=tmp"
  
  cmt br cmt config 
  cmt br make clean $flags
  cmt br make $flags

  cd $iwd
}


av-use-sub(){
  
  local func=autovalidation
  local path=jobs/av 
  
  local jobstring=$(basename $DYW)-$(svnversion $DYW)       # name of the branch or tag and revision number 
  
  local def_cfto="2007-06-19-minimac1"
  local cfto=${1:-$def_cfto}
  
  ## compare_to_date defaults to yesterday in AutoValidate.pl, 
  ## in which case the jobstring is interpreted as a date, so to avoid that
  ## provide the 2nd argument that resets the compareto to a valid 
    

  if [ -d "$DYW_AVOUT" ]; then 

      local cfhist=$DYW_AVOUT/$cfto/$cfto.hists.root
      test -f $cfhist || ( echo cfhist $cfhist does not exist && return 1 ) 

      cd $DYW_AVOUT
      batch-submit $path $func "$jobstring $cfto $@"
      
  else
      echo cannot submit the autovalidation as output folder DYW_AVOUT:[$DYW_AVOUT] doesnt exist ... do av-config first
  fi	  
}

autovalidation(){
   av-use-run $*
}


av-use-run(){

 #
 #  run the local auto validation, comparing with yesterday 
 #
 #  What AutoValidate.pl does :
 #
 #    1) if the jobs have been run already and have created $job.events.root files 
 #       then the job running is skipped ... this allows multiple comparisons
 #
 #    2) loop over jobs running : CreateHistograms.C ... is job local 
 #
 #    3) loop over jobs running : DrawHistograms.C ... has a comparefile argument
 #          $comparefile = "$output_path/$compare_to_date/$compare_to_date.hists.root";
 #
 #
 #  It is hardcoded in "AutoValidate.pl" to compare results from today with those from the passed
 #  day , or default to yesterday ...
 #
 #  SO to do a comparisons more flexibly , change the date on the run
 #  folder to be todays date
 #
 #

    cd $DYW_AVOUT
     
    local today=$(av-use-today)
    local start=${1:-$today} 
    

    [ "$(pwd)" == "$DYW/AutoValidation/scripts" ] && ( echo running autovalidation from $(pwd) is prohibited ) && return

	cnf=av_config.pl
    ok=0	
	if [ -f "$cnf" ]; then
      
      #
      # without "set --" the cmt setup.sh swallows the positional arguments, and complains about them
	  # this removes the positional arguments , as the CMT setup script uses a $* as arguments to cmt 
      # resulting in cmt getting the argumenst inadvertantly
      #       http://tldp.org/LDP/abs/html/internalvariables.html#INCOMPAT
	  #
      local args="$@"
      set -- 
	  source $DYW/G4dyb/cmt/setup.sh

	  xml-env-element
	  xml-llp-element
	  
	  xml-cdata-open
      echo DAYA_DATA_DIR:$DAYA_DATA_DIR
	  local cmd="$DYW/AutoValidation/scripts/AutoValidate.pl $args"
      echo ==== $cmd ==== 
      eval $cmd
      xml-cdata-close		
		
	else   
	  echo $DYW_BASE/av_use.bash:: error config file $cnf doesnt exist
	fi

	rm -f last-av-use-run && ln -s $start last-av-use-run
    
	#  without this get permissioon denied when trying to access the html
	#  from:
    #       http://grid1.phys.ntu.edu.tw:8080/autovalidation/ 
    #
    chmod -R go+rx $start
      
}

av-use-today(){
 # todays date in the av format 2007-04-23	
	perl -MPOSIX -e  "print strftime( '%Y-%m-%d' , localtime(time()) );" 
}


av-use-cf(){

  cd $DYW_AVOUT
  #
  # G1> ln -s 2007-04-20 2007-04-23
  #   fool the script into thinking that we ran the jobs today 
  #

}


av-use-sync(){

  cd $DYW_AVOUT
  local vname=APACHE2_HTDOCS_$SCM_TAG
  eval htdocs=\$$vname
  
  if [ "X$NODE_NAME" == "X" ]; then
     echo ======  av-use-sync ABORTING,  NODE_NAME is not defined on NODE_TAG $NODE_TAG  && return 1 
  fi
  
  if [ "X$htdocs" == "X" ]; then 
     echo ====== av-use-sync destination apache2 instance htdocs on node SCM_TAG $SCM_TAG not setup && return 1
  else
     
     echo ===== creating directory on webserver to store av outputs for node $NODE_NAME
     local cmd1="ssh $SCM_TAG \"mkdir -p $htdocs/autovalidation/$NODE_NAME\" "
     echo $cmd1
     eval $cmd1 
     
     echo ===== rsyncing av outputs to the webserver 
     local cmd2="rsync -e ssh --delete-after -razvt ./ $SCM_TAG:$htdocs/autovalidation/$NODE_NAME/ "
     echo $cmd2
     eval $cmd2
  fi




}



