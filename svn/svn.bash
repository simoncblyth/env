
svn-usage(){
  
   cat << EOU

  For global settings that are above the details of building and
  configuring 

     svn-global-ignores :  
         handholding only
         
EOU

}



#swig-(){         . $ENV_HOME/svn/swig.bash      && swig-env $* ; } 
#svn-apache2-(){  . $ENV_HOME/svn/svn-apache2.bash && svn-apache2-env $* ; }
#svn-sync-(){     . $ENV_HOME/svn/svn-sync.bash  && svn-sync-env  $* ; } 
#svn-tools-(){    . $ENV_HOME/svn/svn-tools.bash && svn-tools-env $* ; }
#svn-build-(){    . $ENV_HOME/svn/svn-build.bash && svn-build-env $* ; } 
#svn-tmp-(){      . $ENV_HOME/svn/svn-tmp.bash   && svn-tmp-env   $* ; } 


svnbuild-(){      . $ENV_HOME/svn/svnbuild/svnbuild.bash   && svnbuild-env   $* ; } 


svn-env(){

  elocal-

  [ "$NODE_APPROACH" == "stock" ] && return 0

  local ver
  case $NODE_TAG in 
    C) ver=1.4.2 ;;
    *) ver=1.4.0 ;;
  esac
  
  export SVN_NAME=subversion-$ver
  export SVN_NAME2=subversion-deps-$ver  
    
  #export SVN_BUILD=$SYSTEM_BASE/svn/build/$SVN_NAME
  export SVN_HOME=$SYSTEM_BASE/svn/$SVN_NAME
  
  export PYTHON_PATH=$SVN_HOME/lib/svn-python:$PYTHON_PATH
  
  svn-path
}


svn-path(){

  local dirs="$SVN_HOME/lib/svn-python/svn $SVN_HOME/lib/svn-python/libsvn"
  for dir in $dirs
  do 
     env-llp-prepend $dir
  done
  

  env-prepend $SVN_HOME/bin
  
}


svn-global-ignores(){

cat << EOI
#  uncomment global-ignores in [miscellany] section of
#     $HOME/.subversion/config
#  setting it to : 
#
global-ignores = setup.sh setup.csh cleanup.sh cleanup.csh Linux-i686* Darwin* InstallArea load.C
#
#    NB there is no whitespace before "global-ignores"
# 
#  after this   
#        svn status -u 
#  should give a short enough report to be useful
#
EOI

echo vi $HOME/.subversion/config


}





