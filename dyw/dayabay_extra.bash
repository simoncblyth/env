
alias p-dayabay-extra="scp $HOME/.bash_dayabay_extra P:"

#
#   usage 
#
#       extra-package-setup      create an additional CMT package $EXTRA_NAME
#
#
#       __extra                  touch main, config, setup, make
#       _extra                   touch main,  setup, make
#
#       extra [macroname]        run ${EXTRA_NAME}App.exe with args: macroname.mac interactive         
#
#       extra-libs               otool/ldd listing of dependent libraries
#
#       vgm-loop [macroname]     loop over detectors running a macro for each              
#
#       viz-loop [macroname]     loop over detectors setting env variable DYW_DETECTOR_SELECT 
#                                and running a macro 
#
#
#


export EXTRA_NAME=G4dybViz
export EXTRA_VERSION=v0
export EXTRA_BRANCH=$EXTRA_NAME/$EXTRA_VERSION
export EXTRA_HOME=$DYW/$EXTRA_BRANCH
export EXTRA_MAIN=dywviz

export VIZ_HOME=$DYW_FOLDER/viz

##	
##   blank version string to give same layout as G4dyb, but then cannot
##   with a complaint about "unstructured versions" on attempting to remove the extra package
##

extra-sync(){

   X=${1:-P}
   vname="DYW_$X"
   eval DYW_X=\$$vname

   if [ "X$DYW_X" == "X" ]; then
	  echo "to syncronise to node $X must set up the $vname variable in .bash_dayabay " 
   else
      echo "syncronise $EXTRA_BRANCH to $X:$vname namely: $X:$DYW_X  ... first a dry run.. then the real thing "
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (  rsync -avn  -e ssh --exclude-from=$DYW/rsync-exclusion-list.txt $DYW/$EXTRA_BRANCH/ $X:$DYW_X/$EXTRA_BRANCH/   ) || echo "cannot extra-sync as LOCAL_NODE=SOURCE_NODE $LOCAL_NODE  " 
      [ "$LOCAL_NODE" == "$SOURCE_NODE" ] && (  rsync -av   -e ssh --exclude-from=$DYW/rsync-exclusion-list.txt $DYW/$EXTRA_BRANCH/ $X:$DYW_X/$EXTRA_BRANCH/   ) || echo "cannot extra-sync as LOCAL_NODE=SOURCE_NODE $LOCAL_NODE  " 
   fi

}


extra-package-setup(){

   ##
   ##   sets up an extra cmt package that "uses" G4dyb package, and populates
   ##   with the main from G4dyb and some dummy others ... 
   ##
   ##   Allows to have a customized main , eg one with the visualization
   ##   setup to produce a standard set of geometry pictures
   ##   this helps to reduce the number of customized files from cvs
   ##
   ##
   ## solved problems :
   ##
   ##    1) failing to pick up the G4dyb includes ????? 
   ##       solved by explicitly #include "G4dyb/whatever.hh" in the code
   ##
   ##    2)  fails to create lib$EXTRA_NAME.so 
   ##       solved by including some dummy code in src and include
   ##

    n=$EXTRA_NAME
	v=$EXTRA_VERSION
    m=$EXTRA_MAIN

    cd $DYW

    if [ -d "$EXTRA_NAME/$EXTRA_VERSION" ]; then
	   echo operation disallowed from safety of existing source $EXTRA_NAME/$EXTRA_VERSION 
    else
	
       cmt remove $n $v            
       cmt create $n $v

       ## copy in the main "dyw.cc" from G4dyb
	   mkdir -p $n/$v/app
       cp -r G4dyb/app/dyw.cc $n/$v/app/$m.cc

       ## copy in the requirements 
	   cp G4dyb/cmt/requirements $n/$v/cmt/

       cd $n/$v/cmt

       echo ------------- editing requirements... from :
	   cat requirements
	   perl -pi -e "s/package G4dyb/package $n/" requirements
	   perl -pi -e 's/(use MCEvent.*)/use G4dyb "*"\n$1/' requirements
    
	   echo ------------  to :
	   cat requirements

       ## modify the main, to explicitly specify the G4dyb in the includes..
       perl -pi -e 's/^(#include\s*\")(dyw.*\".*)$/${1}G4dyb\/$2/'      ../app/$m.cc
       perl -pi -e 's/^(#include\s*\")(LogG4dyb.*\".*)$/${1}G4dyb\/$2/' ../app/$m.cc


       echo ---------- create some dummy code  
	   mkdir -p ../src ../include
	   echo "#include \"Foo.hh\""   > ../src/Foo.cc
	   echo "void Foo(){}"         >> ../src/Foo.cc
       echo "//dummy header "     > ../include/Foo.hh
	   
	 fi  
}




__extra(){
   touch $EXTRA_HOME/app/$EXTRA_MAIN.cc	
   cd $EXTRA_HOME/cmt
   cmt config
   . setup.sh
   cmt make  CMTEXTRATAGS=debug
}


_extra(){
   touch $EXTRA_HOME/app/$EXTRA_MAIN.cc	
   cd $EXTRA_HOME/cmt
   . setup.sh
   cmt make  CMTEXTRATAGS=debug
}



extra(){

    macro=${1:-dybviz}
	cd   $DYM
    cat $macro.mac

	source $EXTRA_HOME/cmt/setup.sh
	#echo doing.... $DYW/InstallArea/$CMTCONFIG/bin/${EXTRA_NAME}App.exe $macro.mac interactive
	#               $DYW/InstallArea/$CMTCONFIG/bin/${EXTRA_NAME}App.exe $macro.mac interactive
    ${EXTRA_NAME}App.exe $macro.mac interactive

}
 

extra-libs(){

	exe=$DYW/InstallArea/$CMTCONFIG/bin/${EXTRA_NAME}App.exe
	if [ "$CMTCONFIG" == "Darwin" ]; then
	    otool -L $exe
	else
	    ldd  $exe
    fi
}




vgm-loop(){


   mkdir -p $VIZ_HOME  && cd $VIZ_HOME

   macro=${1:-vgm-dump}
   app_name=G4dyb
   app_version=.
   tag=root
   mkdir -p detectors

   detectors="Prototype SingleModule SingleModuleAlternative FarSitePool FarSitePool_CD1 ModuleNoVeto NearSitePool NearSitePool_CD1 NearSitePool_CDR FarSite Aberdeen"
   #detectors="Aberdeen"

   for detector in $detectors
   do
	   volume=""
	   outd=$VIZ_HOME/detectors/$detector
	   mkdir -p $outd

	   cd $outd 
	   
	   export DYW_DETECTOR_SELECT=$detector
       export G4DAWNFILE_DEST_DIR="$outd/"   ## dawn needs trailing slash
	   export G4DAWNFILE_VIEWER="dawn -d "   ## -d means dont thow up the GUI, use defaults .DAWN_1.defaults if it exists in the directory 

	   mac=${macro}
	   
	   source $DYW/$app_name/$app_version/cmt/setup.sh
	   echo doing.... $DYW/InstallArea/$CMTCONFIG/bin/${app_name}App.exe $DYM/$mac.mac   tag $tag  detector $detector
	                  $DYW/InstallArea/$CMTCONFIG/bin/${app_name}App.exe $DYM/$mac.mac > ${detector}_${tag}.out 2> ${detector}_${tag}.err
	 

	   ## 2>&1 to put stdout and stderr to the sameplace 
   done

}




viz-loop(){

## 
##  TODO:
##     1) conversions .eps to .pdf .gif
##     2) viewpoints + options , manipulate the .DAWN_1.default file 
##     3) plane sections ... dawncut / david ?
##     4) other format... vrml ? + others ?
##     5) md5 hash... for checking of changes in geometry
##     6) per detector configurable centers etc... 
##
##     7) present results on a web page 
##
##

   macro=${1:-dybviz}

   cd $VIZ_HOME
   mkdir -p detectors

   detectors="Prototype SingleModule SingleModuleAlternative FarSitePool FarSitePool_CD1 ModuleNoVeto NearSitePool NearSitePool_CD1 NearSitePool_CDR FarSite Aberdeen"

   for detector in $detectors
   do
	   volume=""
	   outd=$VIZ_HOME/detectors/$detector
	   mkdir -p $outd

	   cd $outd 
	   
	   export DYW_DETECTOR_SELECT=$detector
       export G4DAWNFILE_DEST_DIR="$outd/"   ## dawn needs trailing slash
	   export G4DAWNFILE_VIEWER="dawn -d "   ## -d means dont thow up the GUI, use defaults .DAWN_1.defaults if it exists in the directory 

	   mac=${macro}_${detector}

       ## hmm is this macro editing now needed following move to using the env to specify
	   ## the detector ?

	   cp -f $DYW_FOLDER/macros/${macro}_template.mac ${mac}.mac
	   perl -pi -e "s/DETECTOR/$detector/" ${mac}.mac
	   perl -pi -e "s/VOLUME/$volume/"     ${mac}.mac
	   
	   source $EXTRA_HOME/cmt/setup.sh
	   echo doing.... $DYW/InstallArea/$CMTCONFIG/bin/${EXTRA_NAME}App.exe $mac.mac
	                  $DYW/InstallArea/$CMTCONFIG/bin/${EXTRA_NAME}App.exe $mac.mac
	  
   done

}



