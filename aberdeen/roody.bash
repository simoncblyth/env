
roody-env(){
   
  export ROODY_NAME=trunk 
  export ROODY_FOLDER=$LOCAL_BASE/roody/$ROODY_NAME

 # alias roody="$ROODY_FOLDER/bin/roody" 

}


roody-path(){

  roody-env
   
  local rbin=$ROODY_FOLDER/bin
  test $PATH == ${PATH/$rbin/} && PATH=$PATH:$rbin

  echo $PATH | tr ":" "\n" 
}


roody-get(){

  roody-env

  local dir=$LOCAL_BASE/roody
  $SUDO mkdir -p $dir && $SUDO chown $USER $dir
  local url=svn://ladd00.triumf.ca/roody/$ROODY_NAME

  cd $dir

  if [ -d "trunk" ]; then
     svn up $url  
  else    
     svn checkout $url
  fi


}


roody-make(){
  roody-env
  
  cd $ROODY_FOLDER
  make

}


