


core-env(){

  cd $LOCAL_BASE
  
  if [ -d "dyb" ]; then
     echo ==== core-env 
  else
     sudo mkdir dyb
     sudo chown $USER dyb     
  fi

  cd dyb

  export DYB_NAME=core-0.0.2
  export SITEROOT=$LOCAL_BASE/dyb/$DYB_NAME
  export DYB_CONF=osx104_ppc_gcc401_dbg  
  
   ## lookup the conf string appropriate to your distro in LCG_Settings/cmt/requirements

}


wget-spoof(){
   echo === spoofing wget with curl ====
   local url=${1}
   local file=$(basename $url)
   local cmd="curl -o $file $url"
   
   echo $cmd
   if [ -f $file ]; then
      echo === wget skipping as already existing $file
   else   
      eval $cmd
   fi    
}

core-get(){

   local name=${1:-$DYB_NAME}
   core-env
   
   local cmd="svn co $DYBSVN/core/tags/$name"
   echo $cmd
   eval $cmd
   
   # see http://dayabay.phys.ntu.edu.tw/tracs/dybsvn/browser/core/tags/core-0.0.2
   # via svn:externals these are fixed revisions of the various parts     
}

core-requirements(){

  core-env
  cd $SITEROOT   # this will be your SITEROOT

  test -d cmt || mkdir cmt 
  
  echo ==== core-requirements  writing cmt/requirements  
  
  
  
  
cat << EOF > cmt/requirements
### requirements ###
# Top level dir holding everything else
set SITEROOT $SITEROOT

# The platform name - this should match the tag that CMT decides for
# you using values in LCG_Settings/cmt/requirements
#set CMTCONFIG "debian_x86_gcc4"
set CMTCONFIG "$DYB_CONF"

# The rest just as is
set CMTEXTRATAGS "dayabay"
path_remove CMTPATH "/gaudi"
path_append CMTPATH "\${SITEROOT}/gaudi"
path_remove CMTPATH "/lcgcmt"
path_append CMTPATH "\${SITEROOT}/lcgcmt"

### end requirements ###
EOF

}