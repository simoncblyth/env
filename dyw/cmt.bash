
[ "$DYW_DBG" == "1" ] && echo cmt.bash

[ "X$CMT_HOME" == "X" ] && ( echo .bash_cmt requires the enviroment setup of .bash_cmt_env including CMT_HOME && return )

cmt-x(){ scp $HOME/$DYW_BASE/cmt.bash ${1:-$TARGET_TAG}:$DYW_BASE ; }

##
##   usage example
##
##        G>  x-cmt         ( transfer just this file to TARGET_TAG node ) 
##        G>  x-cmt P       ( transfer just this file to node P  ) 
##
##        P>  ini           ( pick up the modified environment on target node )
##
##        P>  cmt-use-info  ( list CMT env vars )
##        P>  cmt-get       ( gets the CMT distro and unpacks )
##        P>  cmt-setup     ( compiles CMT )
##       
##



cmt-get(){

   v=$CMT_VERS
   cd $LOCAL_BASE
   test -d cmt || ( $SUDO mkdir cmt && $SUDO chown $USER cmt )

   cd $LOCAL_BASE/cmt
   tgz=CMT$v.tar.gz
   test -f $tgz || curl -o $tgz http://www.cmtsite.org/$v/CMT$v.tar.gz
   test -d CMT/$v || tar zxvf $tgz 
   
}


cmt-setup(){
   cd $CMT_HOME/mgr
   ./INSTALL            ## creates the setup.sh script
   source setup.sh      ## defines CMTROOT / CMTCONFIG CMTBIN and aliases cmt and jcmt 
   env | grep CMT
   make                 ## 
}

cmt-properties(){

   ## for integration into java environment, eg ant builds
   ##
   export CMT_PROPS=$CMT_HOME/cmt.properties
   echo "## these properties are sourced from ~/.bash_cmt " > $CMT_PROPS
   echo "cmt-fold=http://www.cmtsite.org/${CMT_VERS}"       >> $CMT_PROPS
   echo "cmt-dist=CMT${CMT_VERS}"                           >> $CMT_PROPS
}

cmt-get-interfaces(){

   ## recommended by VGM 

  cd $LOCAL_BASE/cmt
  mkdir -p interfaces 
  cd interfaces
  PKGLIST="ROOT Geant4 CLHEP Platform"

## Interfaces/$PKG/vxxx
##
##  cannot find out VXXX ??? 
##  listing cvs modules,
##     http://ximbiot.com/cvs/wiki/index.php?title=CVS_FAQ#Is_there_a_way_to_list_modules_and_files_on_the_server_from_the_client.3F
##  seems to indicate that no such modules exist 
##
  for PKG in $PKGLIST 
  do	  
       echo cvs -d:pserver:anonymous@cvsserver.lal.in2p3.fr:/projects/cvs checkout Interfaces/$PKG/$VXXX
            cvs -d:pserver:anonymous@cvsserver.lal.in2p3.fr:/projects/cvs checkout Interfaces/$PKG/$VXXX
  done
  
}

