tracinit-source(){ echo $BASH_SOURCE ; }
tracinit-usage(){
  cat << EOU
  
    tracinit-source  : $(tracinit-source)
  
    tracinit-prepare name

        CAUTION DO NOT INVOKE DIRECTLTY ... DO SO THRU trac-create 
        FOR EXTRA CHECKING TO AVOID STOMPING ON PREEXISTING INSTANCES

        creates and configures an instance from scratch ... only requiring a 
        pre-existing SVN repository of the same name 
  
    
    
    tracinit-create name
    
        create trac instance, the corresponding SVN repository must already exist
        ... if an instance of that name exists already prompt for confirmation before 
        deleting it and creating a new one 
  
  
    tracinit-newtest 
        test creation + config of a new instance called "newtest"
  
  
EOU

}

tracinit-env(){
  elocal-
}

tracinit--(){
   sudo bash -c "export ENV_HOME=$ENV_HOME ; . $ENV_HOME/env.bash ; trac- ; tracinit- ; $* "
}

tracinit-newtest(){
   tracinit--  tracinit-prepare newtest 
}


tracinit-prepare(){
    
    ## CAUTION IT IS BETTER TO INVOKE THIS THRU trac-create FOR EXTRA CHECKING TO AVOID STOMPING 
    ## PREEXISTING INSTANCES
    
    local msg="=== $FUNCNAME :"
    local name=${1:-newtest}
    [ -z "$name" ] && echo $msg the name of a trac environment is a required argument && return 1


    TRACINIT_CREATE_NOPROMPT=yep tracinit-create $name
    trac-
    trac-configure-instance $name
    trac-inherit-setup

    tracinit-upgrade $name
   
    
    tracperm-
    TRAC_INSTANCE=$name tracperm-prepare
    
    tracinit-logchown $name
    
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
     
     if [ -z "$TRACINIT_CREATE_NOPROMPT" -o "$name" != "newtest" ]; then 
        local answer
        read -p "$msg ARE YOU SURE YOU WANT TO WIPE the \"$name\" trac enviroment from $envp  to allow INITENV ? YES to proceed " answer
        [ "$answer" != "YES" ] &&  echo $msg skipping && return 1
     else
        echo $msg proceeding directly to wipe preexisting instance as TRACINIT_CREATE_NOPROMPT is defined 
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

tracinit-logchown(){
 
    ## suspect the upgrade causes the trac.log to become owned by root ...
    ## causing failure for suibsequrnt apache running
 
    local msg="=== $FUNCNAME :"
    local name=$1
    [ -z "$name" ] && echo $msg the name of a trac environment is a required argument && return 1
    trac-
    local envp=$(trac-envpath $name)
    local user=$(apache- ; apache-user)
    local cmd="$SUDO chown $user:$user $envp/log/$(trac-logname)"
    echo $msg $cmd
    eval $cmd
}




