db2trac-get(){

   cd $LOCAL_BASE/trac
   [ -d "common" ] || mkdir -p common
   cd common

   local name=db2trac
  
   if [ -d "$name" ]; then
      echo db2trac-get ERROR folder $name exists already 
      return 1
   fi
  
   local cmd="svn co http://dayabay.phys.ntu.edu.tw/repos/tracdev/$name/trunk  $name "
   echo $cmd
   eval $cmd    

}


db2trac-update(){

  local name=db2trac
  local dir=$LOCAL_BASE/trac/common/$name
  local cmd="svn update $dir "
  
  if [ -d $dir ]; then
      echo ===== db2trac-update proceededing to: $cmd  
  else
      echo ===== db2trac-update ERROR $dir does not exist 
      return 1 
  fi

  eval $cmd 

}