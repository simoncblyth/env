tracinit-source(){ echo $BASH_SOURCE ; }
tracinit-usage(){
  cat << EOU
  
    tracinit-source  : $(tracinit-source)
  
    tracinit-create name
    
        create trac instance, the corresponding SVN repository must already exist
        ... if an instance of that name exists already prompt for confirmation before 
        deleting it and creating a new one 
  
  
EOU

}

tracinit-env(){
  elocal-
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
     local answer
     read -p "$msg ARE YOU SURE YOU WANT TO WIPE the \"$name\" trac enviroment from $envp  to allow INITENV ? YES to proceed " answer
     [ "$answer" != "YES" ] &&  echo $msg skipping && return 1     
     [ ${#name} -lt 3 ]     && echo $msg name $name is too short not proceeding && return 1
     local dir=$(dirname $envp)
     cd $dir
     [ ! -d "$name" ] && echo $msg repo $name does not exist && return 1
     $SUDO rm -rf "$name"
  fi
           
  local cmd="$SUDO trac-admin $envp initenv $name sqlite:db/trac.db svn $repo "
  echo $msg $cmd
  eval $cmd    

  apache-
  local user=$(apache-user)
  local omd="sudo chown -R $user:$user $envp"
  echo $msg $omd
  eval $omd
  ls -l "$envp" 

  cd $iwd 
}

