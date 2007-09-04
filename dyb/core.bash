
core-env(){

  cd $LOCAL_BASE
  
  if [ -d "dyb" ]; then
     echo ==== core-env 
  else
     sudo mkdir dyb
     sudo chown $USER dyb     
  fi

  cd dyb

}


core-get(){

   core-env
   local cmd="svn co $DYBSVN/core/tags/core-0.0.2"
   echo $cmd
   eval $cmd
   
      
}