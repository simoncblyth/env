

db2trac-dir(){

   if [ "$NODE_TAG" == "G" ]; then
      echo $HOME/db2trac
   else
      echo $LOCAL_BASE/trac/common/db2trac
   fi	  
}

db2trac-get(){

   local dir=$(db2trac-dir)
   local base=$(dirname $dir)
   local name=$(basename $dir)
   local msg="echo $FUNCNAME : "
   
   mkdir -p $base
  
   if [ -d "$name" ]; then
      $msg ERROR folder $name exists already 
      return 1
   fi
  
   local cmd="svn co http://dayabay.phys.ntu.edu.tw/repos/tracdev/$name/trunk  $name "
   echo $cmd
   eval $cmd    

}


db2trac-update(){

  local dir=$(db2trac-dir)
  local cmd="svn update $dir "
  
  if [ -d $dir ]; then
      echo ===== db2trac-update proceededing to: $cmd  
  else
      echo ===== db2trac-update ERROR $dir does not exist 
      return 1 
  fi

  eval $cmd 

}

db2trac-init(){

   local pfx=${FUNCNAME/-init/} 
   local msg="$FUNCNAME hooking up with the checkout ...   "
   
   if [ -f $(db2trac-dir)/$pfx.bash ]; then
      . $(db2trac-dir)/$pfx.bash
   else
      echo $msg cannot proceed as no checkout found
   fi   
   
}