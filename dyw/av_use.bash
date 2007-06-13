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


av-use-sub(){
  
  local func=autovalidation
  local path=jobs/av 

  if [ -d "$DYW_AVOUT" ]; then 

      cd $DYW_AVOUT
      condor-use-submit $path $func "$@"
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
    start=$(av-use-today)

    [ "$(pwd)" == "$DYW/AutoValidation/scripts" ] && ( echo running autovalidation from $(pwd) is prohibited ) && return

	cnf=av_config.pl
    ok=0	
	if [ -f "$cnf" ]; then

	  source $DYW/G4dyb/cmt/setup.sh

	  xml-env-element
	  xml-llp-element
	  
	  xml-cdata-open
      echo DAYA_DATA_DIR:$DAYA_DATA_DIR
	  $DYW/AutoValidation/scripts/AutoValidate.pl
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


