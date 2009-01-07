tracinit-source(){ echo $BASH_SOURCE ; }
tracinit-usage(){
  cat << EOU
  
    tracinit-source  : $(tracinit-source)
  
    tracinit-prepare name
    
        creates and configures an instance from scratch ... only requiring a 
        pre-existing repository of the same name 
  
  
    tracinit-create name
    
        create trac instance, the corresponding SVN repository must already exist
        ... if an instance of that name exists already prompt for confirmation before 
        deleting it and creating a new one 
  
  
EOU

}

tracinit-env(){
  elocal-
}


tracinit-prepare(){
    local msg="=== $FUNCNAME :"
    local name=${1:-newtest}
    [ -z "$name" ] && echo $msg the name of a trac environment is a required argument && return 1

    TRACINIT_CREATE_NOPROMPT=yep tracinit-create $name
    trac-
    trac-configure-instance $name
    trac-inherit-setup


    tracinit-upgrade $name
}


tracinit-create(){

  local iwd=$PWD
  local msg="=== $FUNCNAME :"
  local name=$1
  [ -z "$name" ] && echo $msg the name of a trac environment is a required argument && return 1
  
  
  svn-
  local repo=$(svn-repo-path $name)
  [ ! -d "$repo" ] && echo $msg ABORT no svn repository at repo:$repo && return 1 
  
  trac-
  local envp=$(trac-envpath $name)
  
  if [ -d "$envp" ]; then
     echo $msg a trac environmnent called \"$name\" exists already at $envp 
     ls -l "$envp"
     
     if [ -z "$TRACINIT_CREATE_NOPROMPT" ]; then 
        local answer
        read -p "$msg ARE YOU SURE YOU WANT TO WIPE the \"$name\" trac enviroment from $envp  to allow INITENV ? YES to proceed " answer
        [ "$answer" != "YES" ] &&  echo $msg skipping && return 1
     else
        echo $msg proceeding directly as TRACINIT_CREATE_NOPROMPT is defined 
     fi     
     [ ${#name} -lt 3 ]     && echo $msg name $name is too short not proceeding && return 1
     local dir=$(dirname $envp)
     cd $dir
     [ ! -d "$name" ] && echo $msg repo $name does not exist && return 1
     $SUDO rm -rf "$name"
  fi
           
  python-
  sqlite-
           
  local cmd="$SUDO trac-admin $envp initenv $name sqlite:db/trac.db svn $repo "
  echo $msg $cmd
  eval $cmd    

  apache-
  local user=$(apache-user)
  local omd="$SUDO chown -R $user:$user $envp"
  echo $msg $omd
  eval $omd
  ls -l "$envp" 

  cd $iwd 
}


tracinit-upgrade(){
    local msg="=== $FUNCNAME :"
    local name=$1
    [ -z "$name" ] && echo $msg the name of a trac environment is a required argument && return 1
    trac-
    local envp=$(trac-envpath $name)
    local cmd="$SUDO trac-admin $envp upgrade "
    echo $msg $cmd
    eval $cmd    
}

