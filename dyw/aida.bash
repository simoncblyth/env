#
#   this needs a revisit 
#
#
#
#
#=========================== JAIDA ==========================


aida-usage(){ cat << EOU

EOU
}


[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/aida.bash


export JAIDA_HOME=$LOCAL_BASE/aida/jaida/jaida-3.3.0-1
JAIDA_LOG=$JAIDA_HOME/aida-setup.log
JAIDA_SH=./bin/aida-setup.sh


jaida-setup(){

  ## NB  have to run the setup from $JAIDA_HOME
  ##     many jars are added to the CLASSPATH , a single entry to DYLD..

  echo created by $JAIDA_SH  from $HOME/.bash_aida > $JAIDA_LOG
  date                                            >> $JAIDA_LOG 
  cd $JAIDA_HOME && . $JAIDA_SH && cd             >> $JAIDA_LOG
  echo JAIDA_HOME       : $JAIDA_HOME             >> $JAIDA_LOG
  echo DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH      >> $JAIDA_LOG
  echo CLASSPATH        : $CLASSPATH              >> $JAIDA_LOG

}

#jaida-setup

#========================== AIDAJNI ===========================
# AIDAJNI is a C++ adapter of AIDA - Abstract Interfaces for Data Analysis which
# connects to a Java implementation of AIDA, such as JAIDA.
#
# Originally tried using "AIDAJNI-3.2.3" Darwin distro BUT
# failed to link with the Geant4 example A01 when G4ANALYSIS_USE is set ...
# presumably AIDAJNI was compiled with g++3 rather than g++4 of OSX 10.4
# 
# So rebuilt, as shown below and now use the "-from-src" built version  

#export AIDAJNI_HOME=/usr/local/aida/aidajni/AIDAJNI-3.2.3
export AIDAJNI_HOME=$LOCAL_BASE/aida/aidajni/AIDAJNI-3.2.3-from-src

AIDAJNI_LOG=$AIDAJNI_HOME/aidajni-setup.log
AIDAJNI_SH=./bin/$G4SYSTEM/aidajni-setup.sh

export JDK_HOME=/Library/Java/Home


aidajni-setup(){

  if  test $G4SYSTEM ; then 
     echo created by $AIDAJNI_SH  invoked from $HOME/.bash_aida   > $AIDAJNI_LOG
     date                                                        >> $AIDAJNI_LOG 
     cd $AIDAJNI_HOME && . $AIDAJNI_SH &&  cd                    >> $AIDAJNI_LOG
     echo AIDAJNI_HOME       : $AIDAJNI_HOME                     >> $AIDAJNI_LOG
     echo DYLD_LIBRARY_PATH  : $DYLD_LIBRARY_PATH                >> $AIDAJNI_LOG
     echo CLASSPATH          : $CLASSPATH                        >> $AIDAJNI_LOG
     echo PATH               : $PATH                             >> $AIDAJNI_LOG
     env | grep AIDAJNI                                          >> $AIDAJNI_LOG
     aida-config --version                                       >> $AIDAJNI_LOG
     aida-config --lib                                           >> $AIDAJNI_LOG
     aida-config --include                                       >> $AIDAJNI_LOG
     aida-config --implementation                                >> $AIDAJNI_LOG
  else
     echo G4SYSTEM is not setup, cannot setup AIDAJNI from $HOME/.bash_aida   > $AIDAJNI_LOG
  fi


}

#aidajni-setup

#========================== AIDAJNI-src (build) ===========================
# file:///usr/local/aida/aidajni/AIDAJNI-3.2.3-src/README-AIDAJNI.html
#
#  build aidajni from soure with g++4 on OS X 10.4
#  (remove the below shield to do the build, doesnt work as a macro )
#  this creates the distro :
#       AIDAJNI-3.2.3-Darwin-g++.tar.gz
#   copied this up one level and unpacked and renamed folder to 
#       AIDAJNI-3.2.3-from-src
#
#

shield(){

  export FREEHEP=/usr/local/aida/aidajni/AIDAJNI-3.2.3-src
  export JDK_HOME=/Library/Java/Home
  export PATH=$FREEHEP/bin:$JDK_HOME/bin:$PATH
  #export OS=Darwin
  export OSTYPE=darwin    ## (contrary to documentation) from looking at config/architecture.gmk
  export COMPILER=g++
  cd $FREEHEP
  chmod +x tools/ant
  ./tools/ant -Djar=aidajni
  make  -f GNUmakefile-AIDAJNI 
  make  -f GNUmakefile-AIDAJNI dist 

}

#========================== G4ANALSIS

# export G4ANALYSIS_USE=1

#========================== JAS3

export JAS3_APP=/usr/local/jas3/v0.8.3rc7/jas3-0.8.3/jas3.app
alias jas3="open $JAS3_APP"

#
#  crashes on opening A01.aida 
#    
#Symbol not found: __cg_png_create_read_struct
#  Referenced from:
#  /System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks/ImageIO.framework/Versions/A/ImageIO
#    Expected in: /usr/local/lib/libPng.dylib
#
#    this file doesnt exist but libpng.dylib , does But why is it looking
#    there anyhow ???
#
#   remove /usr/local/lib from DYLD_LIBRARY_PATH  , set in ~/.bash_geant4
#   solves this problem 
#


#==================================

aidalog(){
	echo "---- JAIDA  ------- "
	cat $JAIDA_LOG
	echo "---- AIDAJNI ------- "
	cat $AIDAJNI_LOG
}



