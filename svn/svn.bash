
svn-usage(){
  
   cat << EOU

  For global settings that are above the details of building and
  configuring 

     svn-global-ignores :  
         handholding only
         
     
    
EOU


}


svn-env(){

  elocal-

  [ "$NODE_APPROACH" == "stock" ] && return 0
  
  
  local SVN_NAME=subversion-1.4.0
  local SVN_ABBREV=svn
  
  export SVN_BUILD=$LOCAL_BASE/$SVN_ABBREV/build/$SVN_NAME
  export SVN_HOME=$SYSTEM_BASE/$SVN_ABBREV/$SVN_NAME
  export PYTHON_PATH=$SVN_HOME/lib/svn-python:$PYTHON_PATH
  
  svn-path
  	
}


svn-path(){

  local dirs="$SVN_HOME/lib/svn-python/svn $SVN_HOME/lib/svn-python/libsvn"
  for dir in $dirs
  do 
       if [ "$LOCAL_ARCH" == "Darwin" ]; then 
          export DYLD_LIBRARY_PATH=$dir:$DYLD_LIBRARY_PATH
       else
          export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
       fi
  done
  
  export PATH=$SVN_HOME/bin:$PATH
  
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





