dbi-get(){

 local dir=$LOCAL_BASE/offline/dbi
 mkdir -p $dir 
 
 cd $dir 
 local cmd="svn co $DYBSVN/db/trunk/ "

 echo $cmd
 eval $cmd

}

dbi-update(){

  local dir=$LOCAL_BASE/offline/dbi
  if [! -d "$dir" ]; then
       echo ==== dbi-update ERROR dir $dir does not exist 
  fi

  cd $dir
  svn up 

}
