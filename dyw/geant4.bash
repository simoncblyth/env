
geant4-x(){ scp $HOME/$DYW_BASE/geant4.bash ${1:-$TARGET_TAG}:$DYW_BASE; }

#
#  NB variables beginning G4 are official ones, GQ ones are mine to avoid clashes (Q for Quatre = 4 ) 
#
#   The Geant4 build depends on :
#
#         .bash_clhep       for CLHEP_BASE_DIR .. although it asks for it multiple times anyhow
#         .bash_soxt        for OGLLIBS
#         .bash_openmotif   for XMLIBS and X11LIBS
#
#   NB 
#       on grid1 are attempting to use the /usr supplied motif
#       this could open a can of worms putting /usr/lib in the library path ???
#       ... also time on /d4 is 25 mins behind ... this can cause issues of
#       infinte loops in the builds ... all build items MUST BE ON THE SAME DISK
#
#
#   Recipe for Geant4 build :
#
#     0)  geant4-from-scratch       remove all trace of G4 before rebuilding from tarball
#        OR
#     1)  geant4-wipe               if there is a pre-existing copy of the same name expunge it 
#                                   (a more selective wipe)
#
#     2)  start a new shell        do all following operations in this new shell 
#                                  .bash_geant4_use  will warn that G4 is not setup if
#                                  it has not been commented out of .bash_dyw ( this does not matter  )
#                                  "geant4-off" will comment it out locally    
#
#     3)  geant4-cleanliness       check the environment for G4 env vars
#
#     3.5) set the tag              edit .bash_geant4 to set the desired tag GQ_TAG ,
#                                   for this g4 installation : "dbg" or "bat"
#
#     4)  geant4-get               get and unpack the distro , the tarball will only be downloaded once 
#                                  .. but the unpacking will be redone for different tags GQ_TAG
#
#     5)  geant4-configure         go thru the spanish inquisition creating a config file with the choices    
#                                  ensure to choose: includes , global + granular 
#
#                                ??? i now think global is not wanted had to cd $G4INSTALL/source ; make    
#
#                                  "bat" wrinkles ... use the data dir from the "dbg" tag , to avoid duplicating the large 
#                                   G4 data files , in doing the configure I think need to have G4UI_NODE as n , due to the
#                                   extensive use of macros 
#                                     15:55 .. 17:15  1hr20min
#
#
#     6)  geant4-build             build using the config file created above (eg into $GQ_HOME/{tmp,lib}/Darwin-g++ )
#                                  asks for CLHEP_BASE_DIR again  (presumably as it finds an old one in /usr/local )
#                                  $GQ_HOME/.config/bin/$GQ_SYSTEM/g4make.log                              
#    
#     7)  geant4-not-pedantic      (Darwin only) change the compiler option, then redo geant4-build 
#                                   ..  handles the usual SoBox OpenInventor error  ... on Darwin and x86_64  ??
#
#     8)  geant4-cleanliness       the environment should still be G4 clean at this point, check again
#
#     9)  geant4-make-env          (re)-create the env.{sh,csh} for setting the G4 environment into the pwd which
#                                  is for example $LOCAL_BASE/geant4/$GQ_TAG/$GQ_NAME
#                                  the env.sh script will not override G4WORKDIR if it is set already ...
#                                  otherwise it sets it to $HOME/geant4  ???
#
#    10)  geant4-on                uncomment the ". .bash_geant4_use" in .bash_dyw  and start a new shell, to pick up the environment 
#
#    11)  geant4-vars              should now return a list of 28 or so in "dbg" , 16 in "bat"
#
#    12)  geant4-includes-and-libmap           
#                                  needed for subsequent builds like G4dyb, 
#                                  puts all header files into $GQ_HOME/include/ 
#                                  and creates $G4INSTALL/lib/$G4SYSTEM/libname.map
#
#                                  (huh? also creates dependency files in $G4WORKDIR/tmp/Darwin-g++ ...
#                                  thus ensure this is in appropriately versioned and tagged folder + on same disk (for syncronized times)) 
#
#    13)  geant4-liblist          list the libraries using the liblist program, that was compiled in the previous step
#
#                                 expect order 101 in "dbg" , 
#
#                                 found issue on grid1 : only 20 maybe these are "global"
#                                 if missing libraries, try to build again with :  cd $G4INSTALL/source ; make
#
#    14)  geant4-list-data        list the data files that the get will be downloading
#         geant4-get-data          
#                                 downloads and unpacks the geant4 data files 
#                                 based on the settings in the env.sh file 
#
#    15)  test the geant4 installation , go now to .bash_geant4_use 
#
#    16)  before try G4dyb remember to redo the "dyw-requirements" to account for the new geant4 config and 
#         do an "cmt br cmt config"
#
#
#
#
##### G4 definition ##############################


alias autogui="perl -MSCB::Geant4::Macro -e '&autogui();' "

###########  configure g4 ##################


geant4-vars(){
   env | perl -ne 'm/^G4/&&print' 
}

geant4-cleanliness(){
   ng4=$(geant4-vars | wc -l)
   test $ng4 == "0"  && echo "#the environment is pristine clean  not a G4 in sight " || ( echo "#the environment is G4 dirty ng4=$ng4" && geant4-vars && exit 1  )

   ##
   ##  NB to make the environment G4 pristine, comment out 
   ##      .bash_geant4_use from .bash_dyw
   ##
   ## note on g4pb ... used the ~/.MacOSX/environment.plist to 
   ## set some vars at the GUI level (that includes Terminal.app and iTerm.app and Xcode.app )
   ## did this with my env2gui mechanism , a Cocoa app that propagates a subset of 
   ## the env into the .plist (NB not very convenient as needs a logoff and on to pick up changes )
   ##    grep ENV2GUI $HOME/.bash_*
   ## alias env2gui='/usr/local/osxutils/env2gui/build/Release/env2gui.app/Contents/MacOS/env2gui'
   ##  
   ## see ~/.bash_xcode for more on this
   ##
}


geant4-off(){
  #perl -pi -e 's/^\s(\[.*bash_geant4_use.*)$/#$1/g' $HOME/.bash_dyw
  perl -pi -e 's/^\s(\[.*geant4_use.*)$/#$1/g' $DYW_HOME/dyw_build.bash
}

geant4-on(){
  #perl -pi -e 's/^\#(\[.*bash_geant4_use.*)$/ $1/g' $HOME/.bash_dyw
  perl -pi -e 's/^\#(\[.*geant4_use.*)$/ $1/g' $DYW_HOME/dyw_build.bash
}


geant4-from-scratch(){
  
   chk=do-you-really-want-to-do-this-ctrl-c-if-not
  
   cd $LOCAL_BASE/geant4/$GQ_TAG &&  pwd
   cmd="rm -rvf $GQ_NAME"
   echo $cmd ??????? &&  touch $chk && rm -i $chk
   `$cmd`
   
   
   cd $HOME/geant4/$GQ_NAME &&   pwd
   cmd="rm -rvf tmp"
   echo $cmd ??????? &&  touch $chk && rm -i $chk
   `$cmd`

}

geant4-wipe(){

  cd $GQ_HOME

  echo wiping geant4 distro beneath $GQ_HOME  back to pristine state, 
  echo only the first deletion requires confirmation , so ctrl C out if you dont want to delete
  echo NOTE REBUILDING GEANT4 WILL TAKE AT LEAST A FEW HOURS

  rm -i env.sh 
  rm -f env.sh env.csh config.sh .myconfig.sh env.log
  rm -rf .config
  rm -rf bin lib tmp include
  
}

	
geant4-get(){

  ## this step is sensitive to the GQ_TAG assigned... 

   n=$GQ_NAME

   cd $LOCAL_BASE
   test -d geant4 || ( $SUDO mkdir geant4 && $SUDO chown $USER geant4 )
  
   cd geant4
   tgz=$n.tar.gz
   url=http://geant4.web.cern.ch/geant4/support/source/$tgz



   echo geant4-get tgz $tgz 
   
   test -f $tgz     ||   curl -o $tgz $url 
   test -d $GQ_HOME || ( mkdir -p $GQ_HOME && tar -C $GQ_TAG -zxvf $tgz )
   
}

geant4-configure(){

   
   ngq="$(geant4-vars | wc -l |  perl -pe 's/\s//g' )"
   if [ $ngq == 0 ]; then 

	  echo "# proceed with configure as the environment is G4 clean "
      cd $GQ_HOME

      ## stick into the paste buffer
      if [ "$CMTCONFIG" == "Darwin" ]; then
         echo $CLHEP_BASE_DIR | pbcopy 
	  fi	  

      echo for CLHEP enter CLHEP_BASE_DIR $CLHEP_BASE_DIR
      echo for OpenInventor enter SOXT_HOME $SOXT_HOME
	  echo dont bother with the instructions
   
## go thru the spanish inquisition ... and then exit after creating the config file
##    .config/bin/Linux-g++/config.sh   
      ./Configure -build -E
   
	else
	   echo "# aborting configure as the environment is G4 dirty ngq=$ngq "
	   geant4-cleanliness
    fi
   
}

geant4-build(){
   cd $GQ_HOME
## use the settings from prior inquisition ... and proceed to build   
   ./Configure -d -build 

##  if find curiousities like having to to re-enter the CLHEP_BASE_DIR ...
##  then you probaly started from a non G4 clean  environment OR maybe because
##   the config finds multiple CLHEPs lying around
##
   
}


geant4-list-env(){

   echo OGLLIBS: $OGLLIBS
   echo XMLIBS : $XMLIBS
   echo X11LIBS: $X11LIBS
   echo -------------------
   env | grep G4
   echo -------------------
   env | grep GQ
   echo -------------------
   cat $HOME/env.log

}

geant4-not-pedantic(){

   ##  modify the CXXFLAGS (removing -pedantic) in order to compile G4OpenInventor/SoBox
   ## ./Configure -d  -f oldconfig.sh -build -D CXXFLAGS="-Wall -ansi -Wno-non-virtual-dtor -Wno-long-long"
   ## doesnt work so modify the config file :
   
   perl -pi -e 's/-pedantic//' $GQ_HOME/config/sys/$CMTCONFIG-g++.gmk
}

geant4-make-env(){

   ## creates env.[c]sh , needs to be done after the build 
  	
   cd $GQ_HOME
   ./Configure -e
  
}

geant4-includes-and-libmap(){

   ## this is needed for VGM..  it needs the G4 env setup 
   ## ... it takes a while, checking thru all dependencies
   ##

   if [ "X$G4INSTALL" == "X" ]; then
      echo this needs the G4 environment to be setup, uncomment .bash_geant4_use in .bash_dyw
   else	   
      cd $G4INSTALL/source
      make includes
      make libmap     ## this creates  $G4INSTALL/lib/$G4SYSTEM/libname.map which is now used by G4dyb build system
   fi
}


geant4-liblist(){
  
   ##
   ## huhh gives 100 libs on g4pb, only 20 on grid1 ????
   ##   granular vs global  ???
   ##
   if [ "X$G4INSTALL" == "X" ]; then
      echo this needs the G4 environment to be setup, uncomment .bash_geant4_use in .bash_dyw
   else
       cd $G4INSTALL/lib/$G4SYSTEM
       ./liblist -m . < libname.map  | perl -ne 'print "$_\n" for(split)' | sort 
	   wc libname.map
   fi
}



geant4-parse-env(){
   local g4envsh=$1
   perl -n -e "m|^(\S*)=\"($LOCAL_BASE/.*/data/(.*)(\d\.\d*))\"| && printf \"%s:%s:%s:%s \", \$1,\$3,\$4,\$2 ; "  $g4envsh
}


geant4-list-data(){

   local g4envsh=$G4INSTALL/env.sh 
   echo "geant4-list-data extracting data versions $g4envsh  "   
   
   for quad in $(geant4-parse-env $g4envsh)
      do
	     nvar=`echo $quad | cut -f1 -d:`  
	     base=`echo $quad | cut -f2 -d:`  
	     vers=`echo $quad | cut -f3 -d:`
         path=`echo $quad | cut -f4 -d:`  
		 printf "%-30s %-20s %-10s %s \n" $nvar $base $vers $path
	  done
	  	
}

geant4-get-data(){

   #  http://geant4.web.cern.ch/geant4/support/download.shtml
   #  http://geant4.web.cern.ch/geant4/support/source_archive.shtml
   # the version numbers expected by 481, come from the interactive "./Configure" question session 
   # 
   #  G4NDL.0.2 comes with note [ if thermal cross-sections are not needed ]
   #  thus dont include as covered by G4NDL.3.10
   #
   #
   # data481="G4ELASTIC.1.1 G4NDL.3.9  G4EMLOW.4.0 PhotonEvaporation.2.0 RadiativeDecay.3.0"
   # data482="G4NDL.3.10 G4EMLOW.4.2 PhotonEvaporation.2.0 G4RadioactiveDecay.3.1"
   #   case $GQ_NAME in 
   #     geant4.8.1.p01) data4xx=$data481 ;;
   #         geant4.8.2) data4xx=$data482 ;;
   #                  *) data4xx=error-version-not-handled ;;
   #	esac			
   #
   #
   #
   # avoid hardcoding all these versions and names , by extracting them from the environment file
   # (gives a chance that will work for the next Geant4 version, if the pattern stays the same)
   #
   #  http://geant4.cern.ch/support/source/G4RadioactiveDecay.3.1.tar.gz
   #  

   cd $G4INSTALL
   local g4envsh=$G4INSTALL/env.sh 

   if [ -f "$g4envsh" ]; then

      test -d data || mkdir  data
      cd data
      ## NB generalized data positioning based on the below path , so may not be downloaded to pwd 
   
      for quad in $(geant4-parse-env $g4envsh)
      do
	     nvar=`echo $quad | cut -f1 -d:`  
	     base=`echo $quad | cut -f2 -d:`  
	     vers=`echo $quad | cut -f3 -d:`  
		 path=`echo $quad | cut -f4 -d:`  
           
         ## the unpacked directory name doesnt match the tgz name, hence this shenanigans
         if ([ "$base" == "RadioactiveDecay" ] && [ "$vers" == "3.1" ]) ;then
            prefix="G4" 
         else
            prefix=""
         fi
         
         ##
         ## note the annoying irregular features :
         ##    1) extra "." between base and vers
         ##    2)  irregular prefix in one case 
         ##
          
         tgz=$(dirname $path)/$prefix$base.$vers.tar.gz

	     echo  nvar:$nvar base:$base vers:$vers path:$path tgz:$tgz prefix:$prefix
         
	     test -f "$tgz" && echo already downloaded $tgz || curl -o $tgz http://geant4.web.cern.ch/geant4/support/source/$tgz
	     test -d "$path" &&  echo already unpacked into $path || tar zxvf $tgz
      done

   else
	  echo cannot do geant4-get-data until the geant4 env.sh file has been created 
   fi	   

}



#    clhep issues...
#   checking http://geant4.web.cern.ch/geant4/support/ReleaseNotes4.8.1.html
#   suggests to use ... CLHEP-1.9.2.3
#
#
#Making dependency for file src/G4VMCTruthIO.cc ...
#In file included from include/G4MCTEvent.hh:35,
#                 from include/G4VMCTruthIO.hh:34,
#				  from src/G4VMCTruthIO.cc:33: include/G4MCTGenParticle.hh:34:34:
#								  CLHEP/HepMC/GenEvent.h: No such file or
#								  directory
#								  include/G4MCTGenParticle.hh:35:37:
#								  CLHEP/HepMC/GenParticle.h: No such file or
#								  directory
#								  Making dependency for file src/G4VHepMCIO.cc
#								  ...
#								  In file included from src/G4VHepMCIO.cc:33:
#								  include/G4VHepMCIO.hh:35:34:
#								  CLHEP/HepMC/GenEvent.h: No such file or
#								  directory
#								  Making dependency for file
#								  src/G4VHCIOentry.cc ...
#								  Ma
#
#
#
#
#    
#
#  motif + lib64 issues ...   so rebuild soxt with openmotif 
#
#Creating global shared library
#/usr/local/geant4/geant4.8.1.p01/lib/Linux-g++/libG4interfaces.so ...
#/usr/bin/ld: skipping incompatible /usr/X11R6/lib/libXpm.so when searching for -lXpm
#/usr/bin/ld: skipping incompatible /usr/X11R6/lib/libXpm.a when searching for -lXpm
#/usr/bin/ld: cannot find -lXpm
#collect2: ld returned 1 exit status
#gmake[1]: *** [/usr/local/geant4/geant4.8.1.p01/lib/Linux-g++/libG4interfaces.so] Error 1
#
#Creating shared library
#/usr/local/geant4/geant4.8.1.p01/lib/Linux-g++/libG4FR.so ...
#/usr/bin/ld: cannot find -lSoXt
#collect2: ld returned 1 exit status
#gmake[2]: *** [/usr/local/geant4/geant4.8.1.p01/lib/Linux-g++/libG4FR.so] Error 1
#
#
#
#   sobox issuues....  
#
#Compiling SoBox.cc ...
#src/SoBox.cc:59: error: extra `;'
#gmake[2]: ***
#[/usr/local/geant4/geant4.8.1.p01/tmp/Linux-g++/G4OpenInventor/SoBox.o] Error
#1
#gmake[1]: *** [granular] Error 2
#libc stage done
#
#




