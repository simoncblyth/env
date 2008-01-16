

[ "X$CMTPATH" == "X" ] && echo .bash_dayabay depends on environment settings in .bash_dayabay_use such as CMTPATH which is not set ... ABORT && return
alias ss=". setup.sh"

export DYW_SITE=${DYW}/External/SITE/cmt
export DYW_CMT="$BOOST_CMT $CERNLIB_CMT $CLHEP_CMT $GEANT_CMT $ROOT_CMT $XERCESC_CMT $VGM_CMT"
export ENV2GUI_VARLIST="DYW:$ENV2GUI_VARLIST"


if [ "$LOCAL_NODE" == "g4pb" ]; then
  alias gmake="make"
fi
alias droot="root -l $DYM"   ## cd to $DYM before executing , -l no splash screen


## this config should be done by CMT 
# export DYLD_LIBRARY_PATH=$DYW/InstallArea/$CMTCONFIG/lib:$DYLD_LIBRARY_PATH
# export   LD_LIBRARY_PATH=$DYW/InstallArea/$CMTCONFIG/lib:$LD_LIBRARY_PATH
# export   PATH=$DYW/InstallArea/$CMTCONFIG/bin:$PATH


export GQ_MACPATH=$DYM:$DYW_FOLDER/macros:$DYW/G4dyb:$GQ_MACPATH


## list the bash functions in this file
alias dyw-="grep \(\) $HOME/.bash_dayabay"


dyw-env(){
	
    ## split the piped in string on whitespace and output line by line, 
    ## by virtue of setting the output record separator with the -l

    echo $DYW_CMT | perl -lne 'print for(split)'
}






dyw-get(){

   local defp
   if [ "$DYW_VERSION" == "trunk" ]; then
      defp="legacy/$DYW_VERSION"
   else 
      defp="legacy/branches/$DYW_VERSION" 
   fi
   
   
   local path=${1:-$defp}
   local user=${2:-$USER}
   local branch=$(basename $path)
   local url=$DYBSVN/$path
   
   cd $DYW_FOLDER      
   
   if [ -d "$branch" ]; then
      echo ==== dyw-get ======  a folder called $branch exists already ... skipping checkout , updating instead
      cd $branch
      svn up
   else 
      echo ==== dyw-get ====== proceeding to initial checkout from $url into $branch
      svn --username $user co $url 
      
      cd $branch
      dyw-localize $PWD
   fi

   echo ==== dyw-get completed ====
}


dyw-localize(){

   local iwd=$PWD 
   local base=${1:-$DYW}
   echo ==== dyw-localize base $base ======
   
   # localize the site requirements file , this 
   dyw-requirements $base
  
   # localize geant4 requirements , this changes  External/GEANT/cmt/requirements from a file to a link
   dyw-g4-req $base
  
   #  grid1 fix 
   # dyw-grid1-rootcint-timefix $base

   cd $iwd
   
   echo ==== dyw-localize completed =====
}

#
#  after localization :
#
# [blyth@grid1 legacy-blyth]$ svn st
# ?      External/ROOT/cmt/fragments/rootcint.orig
# M      External/ROOT/cmt/fragments/rootcint
# ?      External/SITE/cmt/requirements
# ?      External/SITE/cmt/requirements.G1
# ?      External/GEANT/cmt/requirements.orig
# M      External/GEANT/cmt/requirements
#
#   THIS CAUSES A PROBLEM FOR HAVING A CLEAN REVISION WHEN WORKING OFF THE REAL REPOSITORY 
#
#



dyw-requirements(){   ## constructs the requirements.$LOCAL_NODE file from $DYW_CMT 

  echo ===== dyw-requirements ===========

  local base=${1:-$DYW}

  cd $base/External/SITE/cmt
  [ "X$NODE_TAG" == "X" ] && echo dyw-requirements error NODE_TAG must be set in env/base/local.bash && return 1

  rex=requirements.example
  #req=requirements.$LOCAL_NODE
  req=requirements.$NODE_TAG
  
  
  
  rm -f $req 

  test -f $req         ||  cp $rex $req
  test -L requirements || ln -s $req  requirements 

  ## the third of the "pair" denotes the cmt type,
  ##  of either a "macro" or "set" definition
  ##  when omitted (as normal) the "macro" type is used

  for pair in $DYW_CMT
  do
	  echo $pair
	  name=`echo $pair | cut -f1 -d:`  
	  valu=`echo $pair | cut -f2 -d:`  
	  type=`echo $pair | cut -f3 -d:`  
	  prefix=${type:-macro}
      echo perl -pi -e "s|^($prefix $name\s*)\"(.*)\"(\s*)$|\$1\"$valu\"\$3|" $req
           perl -pi -e "s|^($prefix $name\s*)\"(.*)\"(\s*)$|\$1\"$valu\"\$3|" $req
  done

  ## potential problems in list looping ... with eg    set OGLLIBS "blah blah"
  ## as the value has spaces...

  perl -pi -e "s|^(set OGLLIBS.*)$|## SCB \$1|" $req 

#  echo "## SCB extras (dyw-requirements)  added to $req " >> $req
#  echo "macro SITE_cppflags \"-fPIC\" " >> $req
#  echo "## SCB end extras  " >> $req


  echo dyw-requirements... compare $req with $rex
  diff $rex $req   

}




    
   

dyw-g4-req(){   ## edits $DYW/External/GEANT/cmt/requirements modifying the "set" sticking in the G4 nominal values to be used

  [ "X$NODE_TAG" == "X" ] && echo dyw-g4-req error NODE_TAG must be set in env/base/local.bash && return 1;
  
  echo === dyw-g4-req start ==========
  
  # check the variables are defined 
  local err=$(base-check-nonzero SOXT_HOME COIN_HOME DYW)
  test ! -z $err && echo $err && return 1 
  
  local base=${1:-$DYW}
  
  cd $base/External/GEANT/cmt
     
  pwd
  g4data_envs="NeutronHPCrossSections G4LEDATA G4LEVELGAMMADATA G4RADIOACTIVEDATA G4ELASTICDATA"
  
  ## only do this the first time ... 
  test -f requirements.orig || cp requirements requirements.orig
  
  ## start from the original , every time  
  rm -f requirements && cp requirements.orig requirements 
  
  ## must remove it otherwise cannot replace with a link 
  #rm -f requirements 
  
  ## follow the site requirements link pattern  

  #local rex=requirements.orig;
  #local req=requirements.$NODE_TAG;
  #rm -f $req;
  #test -f $req || cp $rex $req;
  #test -L requirements || ln -s $req requirements;
  
  # back to original , svn disikes changing a file to a link 
  
  local req=requirements ;
  
  
  for g4data_env in $g4data_envs
  do
	  ## direct reference gives the variable name
	  vname=$g4data_env  
      ## indirect reference to get the value, of these in the environment
      ## http://tldp.org/LDP/abs/html/ivr.html
	  eval vval=\$$g4data_env   
	  echo $vname $vval 
      echo perl -pi -e "s|^(set\s*$vname\s*)(.*)$|\$1 $vval ##(SCB::dyw-g4req) from \$2 |" $req
           perl -pi -e "s|^(set\s*$vname\s*)(.*)$|\$1 $vval ##(SCB::dyw-g4req) from \$2 |" $req 
  done	 




  ## NB vis flags are distinct from UI flags 

  if [ "$CMTCONFIG" == "Darwin" ]; then
    name=DYLD_LIBRARY_PATH
  else
	name=LD_LIBRARY_PATH
  fi
  

  ## the variable GEANT_CMT carries the Geant environment, includeing
  ##      GEANT_ppflags  :  the preprocessor directives 
  ## 

  local GEANT_ppflags
  local GEANT_incdir

  for pair in $GEANT_CMT
  do
	  echo $pair
	  name=`echo $pair | cut -f1 -d:`  
	  valu=`echo $pair | cut -f2 -d:`  
	  type=`echo $pair | cut -f3 -d:`  
 
      if [ "$name" == "GEANT_ppflags" ]; then
          ## replace commas with spaces ... 
         GEANT_ppflags=$(echo $valu | perl -p -e 's/,/ /g')
      elif [ "$name" == "GEANT_incdir" ]; then
         GEANT_incdir=$valu
      fi
 done

  
  local flags=' -I\${GEANT_incdir} ${GEANT_ppflags} '
  
  echo GEANT_CMT     : $GEANT_CMT
  echo GEANT_ppflags : $GEANT_ppflags
  echo GEANT_incdir  : $GEANT_incdir
  echo flags         : $flags
  
  perl -pi -e "s|^(macro\s*GEANT_cppflags\s*\")(.*)(\".*)$|\$1$flags\$3|" $req

  if [ "$GQ_TAG" != "bat" ]; then
    echo # added by dayabay.bash/dyw-g4-req
    echo path_append $name $SOXT_HOME/lib  >> $req 
    echo path_append $name $COIN_HOME/lib  >> $req 
  fi
  diff requirements.orig requirements
  
  echo === dyw-g4-req finished ==========
  
}



dyw-grid1-rootcint-timefix(){

  #
  # this works around an issue of cmt Makefiles going into an infinite loop
  # while build rootcint dictionary dependency files ... 
  # due to non-syncronisation of time stamps on files created by rootcint and
  # those created by touch ... 
  #
  # possible this arises due to non-syncronised times between /tmp and
  # /disk/d4/
  #
  # while investigating this issue made changes to:
  #       DybPolicy/DybPatternPolicy/cmt/cmt_build_deps.sh
  # that dump modification times , using the perl alias ckkmtime from
  #  ~/.bash_perl and ~/perl/SCB/Workflow/PATH.pm
  #
  #
  #  the workaround adds a touch to the rootcint fragment in order that these
  #  generated dictionary files have timestamps in synchrony with other
  #  sources
  #

  local base=${1:-$DYW}   
  local iwd=$PWD
  echo ===== dyw-grid1-rootcint-timefix =====

  cd $base/External/ROOT/cmt/fragments
  
  ## only copy the 1st time
  test -f rootcint.orig || cp rootcint rootcint.orig
  perl -pi -e 's/^(.*rootcint_headers.*\$<\))$/$1 && touch \$@/' rootcint
  diff rootcint rootcint.orig

  cd $iwd
}










dyw-build(){

  local def_type="normal"
  
  local type=${1:-$def_type} 
  local iwd=$PWD
  local branch=$(basename $PWD)

  local flags
  if [ "$GQ_TAG" == "dbg" ]; then
    flags="CMTEXTRATAGS=debug TMP=tmp"
  else
    flags="TMP=tmp" 
  fi  


  local orig_cmtpath=$CMTPATH
  CMTPATH=$DYW_FOLDER/$branch
  BASEPATH=$DYW_FOLDER/$branch  
  
  if [ "$type" == "full" ]; then
  
      echo ===== dyw-build type $type =================
      cd $BASEPATH/DataStructure/MCEvent/cmt
          
      cmt br cmt config 
      cmt br make clean $flags
      cmt br make $flags
  fi
  

  cd $BASEPATH/G4dyb/cmt
  
  echo ==== warning temporary reset of CMTPATH from $orig_cmtpath to $CMTPATH  === PWD $PWD ===
  
  [ -f requirements ] || ( echo ERROR error $PWD with the checkout/update && return 1 ) 

  cmt br cmt config 
  cmt br make clean $flags
  cmt br make $flags



  CMTPATH=$orig_cmtpath  
  echo ==== resetting CMTPATH to $CMTPATH === PWD $PWD

  cd $iwd
}

dyw-rebuild(){

  local flags
  if [ "$GQ_TAG" == "dbg" ]; then
    flags="CMTEXTRATAGS=debug TMP=tmp"
  else
    flags="TMP=tmp" 
  fi  

  cd $DYW/G4dyb/cmt
  cmd="make $flags "
  
  echo ===== dyw-rebuild  $cmd ======
  eval $cmd 

}




dyw-g4dyb-config(){ ##  do a global cmt config  

  local arg=${1}

  echo CMTPATH:[$CMTPATH] arg:[$arg]
  cd $DYW/G4dyb/cmt/

  ## rm -f setup.{sh,csh} cleanup.{sh,csh} Makefile    ## not needed but makes the action of the next step very clear 
  cmt config                                        ## creates the files that were deleted above
  . setup.sh
  
  cmt br cmt config                                 ## configs the dependencies
  . setup.sh 
  
  if [ "$GQ_TAG" == "dbg" ]; then
     echo === building with debug symbols, I hope
     cmt br make $arg CMTEXTRATAGS=debug TMP=tmp
  else
     echo === building without debug symbols
     cmt br make $arg TMP=tmp
  fi

}



#
#    grid1  issue 1  (apr 12-13, 2007) 
#    ------------------------------
#      
#         infinite loops in  in cmt building of dependendies
#         .. investigated using :
#
#               cd $DYW/DataStructure/MCEvent/cmt
#               . setup.sh
#               make -dr 
#
#          see:      
#              dyw-mcevent-infinite-loop
#              dyw-grid1-rootcint-timefix
#              fmtime
#              touch-disks
#
#
# Limitation: Reference member not accessible from the interpreter
#    /disk/d4/dayabay/local/dayabay/dyw_last_20070411/include/Log/LogEntry.hh:23:
#
#------> (Log.make) Rebuilding ../Linux-i686/Log_dependencies.make
#
#------> (Log.make) Rebuilding ../Linux-i686/Log_dependencies.make
#computing dependencies for Log_Dict.cc
#cpp -M  -I"../include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/boost/include/boost-1_33_1"
#-I"/disk/d4/dayabay/local/root/root_v5.14.00b/root/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/External/SITE/src"  -pipe
#-ansi -pedantic -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual
#../Linux-i686/Log_Dict.cc
#------> (Log.make) Rebuilding ../Linux-i686/Log_dependencies.make
#computing dependencies for Log_Dict.cc
#cpp -M  -I"../include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/boost/include/boost-1_33_1"
#-I"/disk/d4/dayabay/local/root/root_v5.14.00b/root/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/External/SITE/src"  -pipe
#-ansi -pedantic -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual
#../Linux-i686/Log_Dict.cc
#
#
#
#    grid1  issue 2  (apr 12-13, 2007) 
#    ------------------------------
#
#          missing libname.map in G4 installation...
#          and missing -lLog ... (==> issue 1 again ?)
#
#    cd $G4INSTALL/source
#    env | grep G4
#    make libmap
#
#   ------> (constituents.make) Building G4dybApp.make
#Application G4dybApp:
#------> (constituents.make) Starting G4dybApp
#../Linux-i686/G4dybApp.exe
#cd ../Linux-i686/; c++   -o G4dybApp.exe.new ../Linux-i686/dyw.o
#-L/disk/d4/dayabay/local/dayabay/dyw_last_20070411/InstallArea/Linux-i686/lib
#-lG4dyb  -lLog  -lUtil    -lMCEvent
#`/disk/d4/dayabay/local/clhep/clhep-1.9.2.3/bin/clhep-config --libs`
#-L/disk/d4/dayabay/local/geant4/dbg/geant4.8.1.p01/lib/Linux-g++
#`/disk/d4/dayabay/local/geant4/dbg/geant4.8.1.p01/lib/Linux-g++/liblist -m
#/disk/d4/dayabay/local/geant4/dbg/geant4.8.1.p01/lib/Linux-g++ <
#/disk/d4/dayabay/local/geant4/dbg/geant4.8.1.p01/lib/Linux-g++/libname.map`
#-L/usr/X11R6/lib -lGLU -lGL
#-L/disk/d4/dayabay/local/xercesc/xerces-c-src_2_7_0/lib -lxerces-c
#-L/disk/d4/dayabay/local/boost/lib
#`/disk/d4/dayabay/local/root/root_v5.14.00b/root/bin/root-config --libs`
#; mv -f G4dybApp.exe.new G4dybApp.exe
#/bin/sh: line 1:
#/disk/d4/dayabay/local/geant4/dbg/geant4.8.1.p01/lib/Linux-g++/libname.map: No
#such file or directory
#/usr/bin/ld: cannot find -lLog
#collect2: ld returned 1 exit status
#mv: cannot stat `G4dybApp.exe.new': No such file or directory
#make[3]: *** [../Linux-i686/G4dybApp.exe] Error 1
#
#




dyw-bmake(){   ## build everything

  cd $DYW/Everything/cmt/
  . setup.sh
  cmt broadcast make
  ##   fails due to lack of CERNLib in Thorium generator

}

#
#   infinite loop in dependency making , maybe ln -s  related
#
#------> (Log.make) Rebuilding ../Linux-i686/Log_dependencies.make
#computing dependencies for Log_Dict.cc
#cpp -M  -I"../include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/include"
#-I"/disk/d4/dayabay/local/boost/include/boost-1_33_1"
#-I"/disk/d4/dayabay/local/root/root_v5.14.00b/root/include"
#-I"/disk/d4/dayabay/local/dayabay/dyw_last_20070411/External/SITE/src"  -pipe
#-ansi -pedantic -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual
#../Linux-i686/Log_Dict.cc
#------> (Log.make) Rebuilding ../Linux-i686/Log_dependencies.make
#


dyw-gen-off(){
  cd $DYW/Everything/cmt 
  echo switch OFF \"use Generators\" in $(pwd)/requirements
  perl -pi -e 's/^.*(use\s*Generators.*)$/#$1/' requirements 
  cat requirements
}

dyw-gen-on(){
  cd $DYW/Everything/cmt 
  echo switch ON \"use Generators\" in $(pwd)/requirements
  perl -pi -e 's/^.*(use\s*Generators.*)$/$1/' requirements 
  cat requirements
}




dyw_(){
   cd $DYW/G4dyb/cmt
   . setup.sh
}


##  cmt experience:
##
##  for a quick change of compiler options... set macro in corres requirements
##        macro cppflags "-pipe -ansi -W -Wall -Wwrite-strings -Wpointer-arith -Woverloaded-virtual"
##  then    
##        cmt config ; make 
##    from the cmt folder
##
## 

dyw-vis-on(){
  cd $DYW/External/GEANT/cmt 
  echo  editing $(pwd)/requirements  ... switching visualisation flags ON ... need to do dyw-config after this
  perl -pi -e 's/^(macro GEANT_cppflags\s*).*$/$1\" -I\${GEANT_incdir} -DG4UI_USE_XM -DG4VIS_USE  -DG4VIS_USE_OPENGLX -DG4VIS_USE_OIX -DG4VIS_USE_OI  \" /' requirements 
  cat requirements 
}

dyw-vis-off(){
  cd $DYW/External/GEANT/cmt 
  echo  editing $(pwd)/requirements  ... switching visualisation flags OFF ... need to do dyw-config after this
  perl -pi -e 's/^(macro GEANT_cppflags\s*).*$/$1\" -I\${GEANT_incdir}   \" /' requirements 
  cat requirements 
}


_mcevent(){

    echo "quick make , for use after MCEvent package code changes only  "
	cd $DYW/DataStructure/MCEvent/cmt
    . setup.sh
	cmt make clean
	cmt make CMTEXTRATAGS=debug

}


___g4dyb(){    ## build G4dyb

#  
#
#  note that i am currently setting preprocessor flags thru the
#  G4dyb/cmt/requirements ... dont do this for G4 flags ... do that in 
#       External/GEANT/cmt/requirements
#
#

    echo "NOTE THAT THIS DEPENDS ON DOING _mcevent AFTER A DATASTRUCTURE CHANGE " 
	echo "SYMPTOM... IS ISSUES WITH dywGLTrack/dywGLEvent etc... "

    echo "very slow make (with config and clean), for use after changing requirements  "

	cd $DYW/G4dyb/cmt
	cmt config               ## i suspect a config is needed following changes to requirements
	. setup.sh
	cmt make clean 
	cmt make CMTEXTRATAGS=debug
}

__g4dyb(){    ## build G4dyb

#  
#
#  note that i am currently setting preprocessor flags thru the
#  G4dyb/cmt/requirements ... dont do this for G4 flags ... do that in 
#       External/GEANT/cmt/requirements
#
#
    echo "slow make (with config but no clean), for use after changing local requirements only .. eg dyw preprocessor flags  "

	cd $DYW/G4dyb/cmt
	cmt config               ## i suspect a config is needed following changes to requirements
	. setup.sh
	cmt make CMTEXTRATAGS=debug
}



_g4dyb(){

    echo "quick make , for use after G4dyb package code changes only  "
	cd $DYW/G4dyb/cmt
	. setup.sh
	cmt make CMTEXTRATAGS=debug

}


g4dyb_(){    ## debug G4dyb.exe  by attaching to the G4dyb.exe process

  ## run this after starting G4dyb.. press c to continue 
  ##  in order to catch crashes
  ## echo GDB attach to G4dyb.exe
  ## `perl -MSCB::Util::Util -e '&gdb_attach("G4dyb.exe");' `

  exe=G4dybApp.exe
  
  pid=`ps wwu | grep $USER | grep $exe | grep -v grep | perl -p -e 's/^\S*\s*(\d*)\s*.*/$1/' `
  echo after continuing with a "c" stop again with a repeated "ctrl-c" 
  `echo gdb + $pid`


}


g4dyb__(){    ## direct executable debugging, sometimes it seems to give more detail than the attach to process technique

    macro=${1:-$DEFAULT_MACRO}
	cd   $DYM
	perl -pi -e 's|^#\s*(/run/beamOn)\s*(\d*)\s*$|$1 1|' $macro.mac
    cat $macro.mac

	source $DYW/G4dyb/cmt/setup.sh
	echo gdb $DYW/InstallArea/$CMTCONFIG/bin/G4dybApp.exe 
	echo enter run then into the gui enter /control/execute $macro.mac 
	gdb $DYW/InstallArea/$CMTCONFIG/bin/G4dybApp.exe 
	

}



x-env-test(){
   ## how to find a remote variable 
   ssh P "bash -lc 'echo DYW:\$DYW'"
}

x-env(){
   X=${1:-$TARGET_TAG}
   vname="DYW_$X"
   eval DYW_X=\$$vname
   echo looking at environment on node $X
   echo ssh $X "bash -lc '. \$DYW/G4dyb/cmt/setup.sh ; env ; ldp '"
        ssh $X "bash -lc '. \$DYW/G4dyb/cmt/setup.sh ; env ; ldp '"
}


g4dyb_env1(){
	cd   $DYM

#  to see in a clean environment
# http://tldp.org/LDP/lfs/LFS-BOOK-6.1.1-HTML/chapter04/settingenvironment.html
# the PS1 seems essential, otherwise the terminal dies
#
#  somehow exec kills the shell ...
#  exec env -i DYW=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ ' /bin/bash
#  exec env -i DYW=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ ' /bin/bash -c "source $DYW/G4dyb/cmt/setup.sh ; env "
#


# just show the pristine controlled environment 
 env -i DYW=$DYW CMTPATH=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ ' /bin/bash -c "env "
    
## cmt setup.sh is sourcing .bash_profile !!! and get whining as 
# sh: line 1: system_profiler: command not found
# #CMT> Warning: apply_tag with empty name [$(cmt_system_version)]
# sh: line 1: system_profiler: command not found
# sh: line 1: system_profiler: command not found
# sh: line 1: system_profiler: command not found
#
env -i DYW=$DYW CMTPATH=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ ' /bin/bash -c "source $DYW/G4dyb/cmt/setup.sh ; env "

## try to isolate this bad behaviour ... not here 
env -i DYW=$DYW CMTPATH=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ ' CMTROOT=$CMTROOT /bin/bash -c ". $CMTROOT/mgr/setup.sh ; env "

## not here
env -i DYW=$DYW CMTPATH=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ ' CMTROOT=$CMTROOT /bin/bash -c ". $CMTROOT/mgr/setup.sh ; tempfile=`${CMTROOT}/mgr/cmt -quiet build temporary_name` ; echo $tempfile ; env "  

## culprit identified   ${CMTROOT}/mgr/cmt  has a naked /bin/sh shebang line  
env -i DYW=$DYW CMTPATH=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ ' CMTROOT=$CMTROOT /bin/bash -c ". $CMTROOT/mgr/setup.sh ; tempfile=`${CMTROOT}/mgr/cmt -quiet build temporary_name` ; ${CMTROOT}/mgr/cmt setup -sh -pack=G4dyb -version=v0 -path=/Users/blyth/Work/dayabay/geant4.8.2.p01/dbg/dyw_release_2_9_wc  -no_cleanup $* >${tempfile}; cat ${tempfile} ; env "
  	
}



g4dyb-pristine-env(){
  
   ## added display as its needed in Xcode for debugging with G4UIXm active to work 
  env -i DYW=$DYW CMTPATH=$DYW HOME=$HOME TERM=$TERM PS1='\u:\w\$ '  DISPLAY=:0.0 /bin/bash -c "source $DYW/G4dyb/cmt/setup.sh ; env " 

}



g4dyb_s(){    ## run G4dyb in session mode, that is with the G4UIXm interface ... advantage is that can attach the debugger prior to launch 

    macro=${1:-$DEFAULT_MACRO}
	cd   $DYW_FOLDER/macros

    source $DYW/G4dyb/cmt/setup.sh
	
    ## CMT sets up the path, so dont need to specify here
	## hmm howabout different Geant4 compilation tags : "dbg" and "pro" propagation to here ??
	## $DYW/InstallArea/$CMTCONFIG/bin/G4dybApp.exe
    ##
	
	which G4dybApp.exe
    G4dybApp.exe
	
}

g4dyb_i(){    ## run G4dyb in interactive mode

    macro=${1:-$DEFAULT_MACRO}

	cd   $DYM
	perl -pi -e 's|^(/run/beamOn.*)$|# $1|' $macro.mac
	
	source $DYW/G4dyb/cmt/setup.sh
	$DYW/InstallArea/$CMTCONFIG/bin/G4dybApp.exe $macro.mac interactive
}




g4dyb-momo(){   ## java interface "momo" experiments ... not working with open inventor

##
##  provides java based gui to control a G4 app ... requires use of
##  G4UIGAG for the session rather than G4UIterminal or G4UIXm
##
##  seems to not work with OpenInventor... get a white screen with no widgets
##

  source $DYW/G4dyb/cmt/setup.sh
  cd $DYW/G4dyb/data
  echo DAYA_DATA_DIR $DAYA_DATA_DIR
  export MOMO_HOME=${GQ_HOME}/environments/MOMO 
  java -cp ${MOMO_HOME}/MOMO.jar momo

}










## FIXED ? issues ...
##
#======  Couldnt find cernlib...
#
#[pal] /usr/local/cernlib/2005 > ln -s slc4_amd64_gcc34 pro
#
#
# ========   Couldnt make shared lib as not PIC ...
#
#
#   added to Everything/SITE/requirements ... via dyw-requirements
#      macro  SITE_cppflags -fPIC
#
#
# =========  unittest issue ... cant find boost headers
#
#	     arghhh, into BOOST_INAME in .bash_boost  "-" rather than "_"
#
#
#=========  fortran Thorium issue
#
#		    comment out use Generators for now...
#        fixed by changing paths in CERNLIB_CMT , see .bash_cernlib
#
#
#
#========  compilation issue  with rootcint and namespaces, using
#			root_v5.14.00b.source.tar.gz
#
#			   cd ../amd64_linux26/; c++ -c -I"../include"
#			   -I"/home/sblyth/Work/dayabay/dywcvs/include"
#			   -I"/home/sblyth/Work/dayabay/dywcvs/include"
#			   -I"/home/sblyth/Work/dayabay/dywcvs/include"-I"/usr/local/boost/include/boost-1_33_1"
#			   -I"/usr/local/root/root/include"
#			   -I"/home/sblyth/Work/dayabay/dywcvs/External/SITE/src"   -pipe
#			   -ansi -pedantic -W -Wall -Wwrite-strings -W-Woverloaded-virtual
#			   -D_GNU_SOURCE -o Log_Dict.o
#			   `/usr/local/root/root/bin/root-config --cflags` -Wno-long-long
#			   -fPIC        -I../amd64_linux26
#			   ../amd64_linux26/Log_Dict.cc
#			   ../amd64_linux26/Log_Dict.cc: In function `int
#			   G__Log_Dict_247_0_1(G__value*, const char*, G__param*, int)':
#			   ../amd64_linux26/Log_Dict.cc:2084: error: `Logging::fptr'
#			   should have been declared inside `Logging'
#			   make[3]: *** [../amd64_linux26/Log_Dict.o] Error 1
#			   make[2]: *** [Log] Error 2
#			   make[1]: *** [common_target] Error 2
#			   make: *** [check_config] Error 2
#			   CMT> Error: execution_error : make
#
#		
#   kludge this by adding some _Dict.cc editing to :   External/ROOT/cmt/fragments/rootcint
#         echo SCB kludge exrera $@
#         perl -pi -e 's/Logging::\*fptr/\*fptr/' $@
#
#
# =========  thence linking incompatible libGL  issue 
#   comment out the set OGLLIBS in requirements.pal 
#
#
# 
# cd ../amd64_linux26/; c++   -o G4dyb.exe.new ../amd64_linux26/dywCerenkov.o
# ../amd64_linux26/GLG4PosGen.o ../amd64_linux26/dywRPC_SD.o
# ../amd64_linux26/dywEllipsoid.o ../amd64_linux26/LogG4dyb.o
# ../amd64_linux26/dywTrackingAction.o ../amd64_linux26/dyw_PMT_LogicalVolume.o
# ../amd64_linux26/dywConstructAberdeenLab.o
# ../amd64_linux26/dywConstructFarSitePool.o
# ../amd64_linux26/dywabPlasticScintHit.o ../amd64_linux26/dywGdCaptureGammas.o
# ../amd64_linux26/dywRunAction.o ../amd64_linux26/dywScintHit.o
# ../amd64_linux26/dywConstructTunnelLab.o ../amd64_linux26/dywScintSD.o
# ../amd64_linux26/dywabPlasticScintSD.o ../amd64_linux26/dywPhysicsList.o
# ../amd64_linux26/dywPrimaryGeneratorAction.o
# ../amd64_linux26/dywPhysicsListMessenger.o ../amd64_linux26/dywUtilities.o
# ../amd64_linux26/dywConstructNearSitePool_CDR.o
# ../amd64_linux26/dywAnalysisManager.o
# ../amd64_linux26/dywDOMTreeErrorReporter.o ../amd64_linux26/GLG4VertexGen.o
# ../amd64_linux26/dywTrackInformation.o
# ../amd64_linux26/dywGeneratorMessenger.o ../amd64_linux26/dywHit_PMT.o
# ../amd64_linux26/dywabPropTubeHit.o ../amd64_linux26/dywDetectorMessenger.o
# ../amd64_linux26/dywConstructDYBDetector.o ../amd64_linux26/dywXMLReader.o
# ../amd64_linux26/dywStackingAction.o
# ../amd64_linux26/dywConstructOneModuleAlternative.o
# ../amd64_linux26/dywPMTOpticalModel.o ../amd64_linux26/dywEventAction.o
# ../amd64_linux26/dywDetectorConstruction.o ../amd64_linux26/dywGenerator2.o
# ../amd64_linux26/dywConstructPrototype.o
# ../amd64_linux26/dywConstructOneModule.o ../amd64_linux26/dywScintillation.o
# ../amd64_linux26/dywSD_PMT.o ../amd64_linux26/dywRPCLogicalVolume.o
# ../amd64_linux26/dywConstructNearSitePool.o ../amd64_linux26/dywRPC_Hit.o
# ../amd64_linux26/dywGdNeutronHPCaptureFS.o ../amd64_linux26/dywTorusStack.o
# ../amd64_linux26/dywNeutronHPCapture.o ../amd64_linux26/dywSteppingAction.o
# ../amd64_linux26/dywabPropTubeSD.o ../amd64_linux26/dywConstructFarSite.o
# ../amd64_linux26/dywConstructModuleNoVeto.o ../amd64_linux26/dyw.o
# -L/home/sblyth/Work/dayabay/dywcvs/InstallArea/amd64_linux26/lib      -lLog
# -lUtil    -lMCEvent    `/usr/local/clhep/clhep-1.9.2.3/bin/clhep-config
# --libs`  `/home/sblyth/Work/dayabay/dywcvs/External/GEANT/cmt/geant4-config
# -ldflags` -lGLU -lGL -L/usr/X11R6/lib -lXmu
# -L/usr/local/xercesc/xerces-c-src_2_7_0/lib -lxerces-c  -L/usr/local/boost/lib
# `/usr/local/root/root/bin/root-config --libs`      ; mv -f G4dyb.exe.new
# G4dyb.exe
# /usr/bin/ld: skipping incompatible /usr/X11R6/lib/libGLU.so when searching for
# -lGLU
# /usr/bin/ld: skipping incompatible /usr/X11R6/lib/libGLU.a when searching for
# -lGLU
# /usr/bin/ld: skipping incompatible /usr/X11R6/lib/libGL.so when searching for
# -lGL
# /usr/bin/ld: skipping incompatible /usr/X11R6/lib/libGL.a when searching for
# -lGL
# /usr/bin/ld: skipping incompatible /usr/X11R6/lib/libXmu.so when searching for
# -lXmu
# /usr/bin/ld: skipping incompatible /usr/X11R6/lib/libXmu.a when searching for
# -lXmu
# /usr/bin/ld: cannot find -lXmu
# collect2: ld returned 1 exit status
# mv: cannot stat `G4dyb.exe.new': No such file or directory
# make[3]: *** [../amd64_linux26/G4dyb.exe] Error 1
# 
#
#  ======== running G4dyb issue 1 ... cannot find G4EMLOW3.0/smth
#
#  searching requirements reveals a "set" of G4EMLOW3.0
#  [pal] /home/sblyth/Work/dayabay/dywcvs > dyw-rf G4EMLO                     
# /home/sblyth/Work/dayabay/dywcvs/External/GEANT/cmt/requirements:set G4LEDATA "${G4DATA}/G4EMLOW3.0"
#
#    try to solve by unsetting these... thence should  use the Geant4 defaults
#
# set NeutronHPCrossSections "${G4DATA}/G4NDL3.8"
# set G4LEDATA "${G4DATA}/G4EMLOW3.0"
# set G4LEVELGAMMADATA "${G4DATA}/PhotonEvaporation2.0"
# set G4RADIOACTIVEDATA "${G4DATA}/RadiativeDecay3.0"
# set G4ELASTICDATA "${G4DATA}/G4ELASTIC1.1"
#
#
#
#[dyb]   =W= Warning: setting PMT mirror reflectivity to 0.9999 because no PMT_Mirror material properties defined
# [geant] =W= G4EMDataSet::LoadData - data file "/usr/local/geant4/geant4.8.1.p01/data/G4EMLOW3.0/rayl/re-ff-1.dat" not found
#[geant] =W= 
#[geant] =W= *** G4Exception: Aborting execution ***
#Aborted
#[pal] /home/sblyth/Work/dayabay/dywcvs/G4dyb/data > 
#
#
#--------------------------------------------------------------
# Now trying [make] in /Users/blyth/Work/dayabay/dywcvs/Test/TestApp/cmt
## (14/17)
##--------------------------------------------------------------
#------> (Makefile.header) Rebuilding constituents.make
#------> (constituents.make) Rebuilding setup.make Darwin.make
#CMTCONFIG=Darwin
#setup.make ok
#Darwin.make ok
#------> (constituents.make) Rebuilding library links
#------> (constituents.make) all done
#------> (constituents.make) Building TestApp.make
#Application TestApp
#------> (constituents.make) Starting TestApp
#------> (TestApp.make) Rebuilding ../Darwin/TestApp_dependencies.make
#computing dependencies for UnitTestHookin.cc
#cpp -M  -I"../include" -I"/Users/blyth/Work/dayabay/dywcvs/include"
#-I"/Users/blyth/Work/dayabay/dywcvs/include"
#-I"/Users/blyth/Work/dayabay/dywcvs/include"
#-I"/usr/local/boost/include/boost-1_33_1"
#-I"/Users/blyth/Work/dayabay/dywcvs/External/SITE/src"
#/Users/blyth/Work/dayabay/dywcvs/Test/TestApp/app/UnitTestHookin.cc
#../Darwin/UnitTestHookin.o
#cd ../Darwin/; c++ -c -I"../include"
#-I"/Users/blyth/Work/dayabay/dywcvs/include"
#-I"/Users/blyth/Work/dayabay/dywcvs/include"
#-I"/Users/blyth/Work/dayabay/dywcvs/include"
#-I"/usr/local/boost/include/boost-1_33_1"
#-I"/Users/blyth/Work/dayabay/dywcvs/External/SITE/src"     -o UnitTestHookin.o
#-I/Users/blyth/Work/dayabay/dywcvs/Test/TestApp/app
#/Users/blyth/Work/dayabay/dywcvs/Test/TestApp/app/UnitTestHookin.cc
#../Darwin/TestApp.exe
#cd ../Darwin/; c++   -o TestApp.exe.new ../Darwin/UnitTestHookin.o
#-L/Users/blyth/Work/dayabay/dywcvs/InstallArea/Darwin/lib    -lUtilUnitTest
#-lUnitTest   -lUtil      -L/usr/local/boost/lib
#-lboost_unit_test_framework-gcc  -lboost_program_options-gcc       ; mv -f
#TestApp.exe.new TestApp.exe
#/usr/bin/ld: can't locate file for: -lboost_unit_test_framework-gcc
#collect2: ld returned 1 exit status
#mv: rename TestApp.exe.new to TestApp.exe: No such file or directory
#make[3]: *** [../Darwin/TestApp.exe] Error 1
#make[2]: *** [TestApp] Error 2
#make[1]: *** [common_target] Error 2
#make: *** [check_config] Error 2
#CMT> Error: execution_error : make
#
#
#     incorrect BOOST_LIBTYPE for Darwin, see .bash_boost 
#
#
#    order 8 GB 
#
# [g4pb:/usr/local] blyth$ du -hs xercesc vgm soxt root openmotif graphics
# geant4 coin3d cmt clhep cernlib boost aida
# 137M    xercesc
#  60M    vgm
#  8.1M    soxt
#  543M    root
#   36M    openmotif
#   2.6G    graphics
#   2.1G    geant4
#    62M    coin3d
#    9.5M    cmt
#    629M    clhep
#    524M    cernlib
#    576M    boost
#     48M    aida
# 	[g4pb:/usr/local] blyth$ l graphics/
# 	total 21016
# 	drwxr-xr-x   45 blyth  wheel     1530 Feb  6 18:02 Coin-2.4.5
# 	drwxr-xr-x   19 blyth  wheel      646 Feb  6 18:31 Coin-2.4.5-build
# 	-rw-r--r--    1 blyth  wheel  4721898 Feb  6 17:13 Coin-2.4.5.tar.gz
# 	drwxr-xr-x   17 blyth  blyth      578 Feb  6 17:57 Coin3d-dev
# 	drwxr-xr-x    5 blyth  blyth      170 Dec 25 19:15 OpenInventor
# 	drwxr-xr-x    5 blyth  blyth      170 Dec 22 16:46 Qt-mac
# 	drwxr-xr-x   30 blyth  wheel     1020 Feb  6 18:58 SoXt-1.2.2
# 	-rw-r--r--    1 blyth  wheel   883830 Feb  6 18:42 SoXt-1.2.2.tar.gz
# 	drwxr-xr-x   40 blyth  wheel     1360 Jan 30 17:30 openMotif-2.2.3
# 	-rw-r--r--    1 blyth  blyth  5149785 Apr  7  2004 openMotif-2.2.3.tar.gz
# 
# 
# 
# 
# 
# 
