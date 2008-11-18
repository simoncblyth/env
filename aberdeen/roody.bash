

roody-usage(){
  cat << EOU
  
     roody-version : $(roody-version)
     roody-base    : $(roody-base)
     roody-folder  : $(roody-folder)
     roody-url     : $(roody-url)
     
     
     ROOTSYS : $ROOTSYS
     
     roody-cd 
     roody-get

  
EOU

}


roody-cd(){
  cd $(roody-folder)
}


roody-env(){
   
  local- 

  root-
  roody-path

 # alias roody="$ROODY_FOLDER/bin/roody" 

}

roody-version(){
   case $NODE_TAG in 
     *) echo trunk ;;
   esac  
}

roody-base(){
  case $NODE_TAG in 
     *) echo $LOCAL_BASE/env/roody ;;
  esac   
}

roody-folder(){ echo $(roody-base)/$(roody-version)  ; }

roody-url(){
 #local url=svn://ladd00.triumf.ca/roody/$ROODY_NAME
  echo http://dayabay.phys.ntu.edu.tw/repos/aberdeen/$(roody-version)/roody
}



roody-path(){
  local rbin=$(roody-folder)/bin
  if [ -d "$rbin" ]; then
     test $PATH == ${PATH/$rbin/} && PATH=$PATH:$rbin
  fi
  #echo $PATH | tr ":" "\n"
   
}


roody-get(){

  local dir=$(dirname $(roody-folder))
  $SUDO mkdir -p $dir && $SUDO chown $USER $dir
 
  local url=$(roody-url)
  local ver=$(roody-version)

  cd $dir
   
  if [ ! -d "$ver" ]; then 
     svn checkout $url $ver 
  else
     cd $ver
     svn up   
  fi

}


roody-make(){
  
  
  
  cd $(roody-folder)
  make

}


