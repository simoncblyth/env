##  ================ Recipe to perform auto validation ======================
##
##  P>  av-config ## creates the config file specifying the date to compare to and output directory 
##                          ... having nothing to compare with doesnt cause failure
##
##  P>  av-sub        condor submission of the autovalidation ...
##                  ... failing with 
##
##          007 (9520.000.000) 04/20 15:29:51 Shadow exception!
##          Can no longer talk to condor_starter on execute machine (192.168.3.51)
##		                0  -  Run Bytes Sent By Job
##				        0  -  Run Bytes Received By Job
##
##  G>  
##
##
##
##       av-run          ## run the jobs , takes a long time
##
##
##       av-getref      ## get reference hists to allow a comparison to be made
##
##       av-run          ## run again, is clever enough not to rerun the jobs, and
##                          will make comparisons this time ... if the relevant reference files have been
##                          grabbed
##
##       av-getout       ## copy the outputs into a local webserver, for easy browsing
##                       
##
##   ================ looking at AutoValidate scripts... =====================
##
##
##   AutoValidate.pl 
##                     defaults to compare with yesterday, can specify another
##                     reference date as argument
##
##   CreateHistograms.C    
##                       jobname
##         from  {gamma_1Mev,eminus_1MeV,..}.event.root projecting 
##         into jobnamed subdirectories of tree $datestring.hists.root
##
##
##    DrawHistograms.C
##          takes as input the paths to this .hists.root file and a reference
##          such file, creates jobname.summary (tab delimited) files that contain the histo
##          means and widths from this run and the reference ... greates .{gif,pdf}
##
##
##
##


av-x(){ scp  $HOME/$DYW_BASE/av.bash ${1:-$TARGET_TAG}:$DYW_BASE/av.bash ; }

av-config(){

    # create the NODE specifiv AutoValidation config file and creates 
    # a link to point to it 
    # ... config includes the output directory 

    #cd $DYW/AutoValidation/scripts 
	#cnf=av_config.pl

    test -d $DYW_AVOUT || mkdir -p $DYW_AVOUT 
	cd $DYW_AVOUT

    [ "$(pwd)" == "$DYW/AutoValidation/scripts" ] && ( echo running autovalidation from $(pwd) is prohibited ) && return

	local cnf=av_config.pl
    rm -f $cnf   

    echo ========== av-config writing cnf:$cnf 

    cat << EOC > $cnf
#
# Configuration file for AutoValidator
# 
#\$output_path="\$ENV{CMTPATH}/AutoValidation/output";
\$output_path="$DYW_AVOUT";
\$compare_to_date="yesterday";
\$replace=1; # 1=replace data, 0=make new version.
#\$url="http://minimac1.phy.tufts.edu/~tagg/AutoValidation/";
#\$url="";
\$url="http://grid1.phys.ntu.edu.tw:8080/autovalidation";
## unfortunately AutoValidate.pl does not utilize the advantages of relative
## links .. so have to go absolute to comply
##
\$mailfrom='AutoValidator <sblyth@nuu.edu.tw>';
#\$mailto='Simulation Group <theta13-simulation@theta13.lbl.gov>';
\$mailto='Me <blyth@hep1.phys.ntu.edu.tw>';
\$domail="no";

@jobs=( "gamma_1MeV", "eminus_1MeV", "neutron_1keV", "muon_100GeV");

# Config file must end with true expression, to show it loaded correctly.
1;
   
EOC

   cat $cnf 

   ##rm -f $cnf &&  ln -s $cnfn $cnf  


}


av-yesterday(){
	
	if [ "$CMTCONFIG" == "Darwin" ] ; then
      timdef=$(perl -e 'print time-(60*60*24)')
	  refdef=$(date -r $timdef +"%Y-%m-%d")  
    else		
	  refdef=$(date -d yesterday +"%Y-%m-%d")
	fi  
	echo $refdef 
}    

av-getref(){

    ##
    ##  grab reference root files for comparison with local ones..
    ##   usage 
    ##       av-getref 2007-02-07
    ##    (defaults to yesterday .. which is may be available )
    ##


    local cnf="$DYW/AutoValidation/scripts/av_config.pl"
    
    if [ -f "$cnf" ]; then
       echo using cnf:$cnf
    else
       echo ERROR cnf:$cnf not found, you need to av-config first 
       return 1
    fi   


    local refav=http://minimac1.phy.tufts.edu/~tagg/AutoValidation
    local refdef=$(av-yesterday)
	local refday=${1:-$refdef}
    local refur=$refav/$refday

	echo ====== av-getref === attempt to grab the reference files from day $refday , default is $refdef 
    
	test -d $DYW_AVOUT || mkdir -p $DYW_AVOUT
	cd $DYW_AVOUT

    test -d $refday || mkdir $refday
	ln -s $refday last-av-getref
	cd $refday

    ## this extracts the jobs list that was entered above, by the av-setup
    macs=`perl -e "require "$cnf" ; print \"\@jobs\" ; "`
    typs="events.root log summary gif pdf" 

    echo jobs: $macs pwd:$PWD

    for mac in $macs
    do 
       for typ in $typs 
       do 	
	      evfil=$mac.$typ
	      evurl=$refur/$evfil
		  echo $evurl $evfil
	      test -f $evfil && echo file $evfil exists already  || curl -o $evfil $evurl
       done
    done	   

    ## this one contains all the hists from the jobs 
    hisfil=$refday.hists.root
	hisurl=$refur/$hisfil
	echo $hisurl $hisfil
    test -f $hisfil && echo file $hisfil exists already || curl -o $hisfil $hisurl
	
}




av-getout(){
  
  LOCAL_MACHINE=$(uname -a | cut -d " " -f 2 ) 
  wbase=$APACHE_HTDOCS

  today=$(date +"%Y-%m-%d")	
  #avdays="2007-02-08 2007-02-07 2007-02-06 2007-02-05"
  avdays="$today"

  if [ "$LOCAL_NODE" == "g4pb" ]; then
	  
     scp P:$DYW_AVOUT/index.html $wbase/autovalidation.html
     for avday in $avdays
     do	  
	    test -d $wbase/$avday ||  scp -r P:$DYW_AVOUT/$avday $wbase 
        open http://$LOCAL_MACHINE/$avday/result.html
     done
     open http://$LOCAL_MACHINE/autovalidation.html

  else

     $SUDO cp -f $DYW_AVOUT/index.html $wbase/autovalidation.html
     for avday in $avdays
     do	  
         test -d $wbase/$avday || $SUDO cp -rf $DYW_AVOUT/$avday $wbase   		  
     done
     echo see results at: http://$LOCAL_MACHINE:8080/autovalidation.html

  fi



}


