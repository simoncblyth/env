private-src(){  echo base/private.bash ; }
private-source(){ echo ${BASH_SOURCE:-$(env-home)/$(private-src)} ; }
private-vi(){     vi $(private-source) ; }
private-usage(){

   cat << EOU
  
   ----
       issue 1 

          the webserver apache user needs 
          to read this file ... but it must be protected 
          so when need to use it for testing get permission denied

          -rw-------  1 apache apache 452 May 29 12:48 /data/env/local/env/.bash_private

          workaround is to duplicate the path and have different locations for
          different users ... this needs simplification 


   ----

      The private file is by default located at 
           $ENV_HOME/../.bash_private

      This can be overridden with the envvar ENV_PRIVATE_PATH 


    

       Access the values with 
            private-
            private-val PRIVATE_VARIABLE <path>

       alternate access, that works when needs a sudo password
            private-get PRIVATE_VARIABLE <path>
  

       Export a list of variables into envvars with 
           private-export VAR1 VAR2




       Change, detect or add values with :
            private-hasval- NAME
            private-set NAME VAL

       TODO: detect permissions on the path 
             to avoid pointless sudo 
               



 
       The path defaults to  $(private-path)     
       The definition file should have variables defined as below :
           local PRIVATE_VARIABLE=42
      
      (this is more secure than exporting them)
      
       returns blank if not defined or if the permissions on the file are 
       not "-rw-------"
       
       returns silently if the file doesnt exist 


       Arc> private-test- | sudo python
                    i = p.rfind('/') + 1
                    AttributeError: 'NoneType' object has no attribute 'rfind'
                    
       Arc> private-test- | sudo ENV_PRIVATE_PATH=$HOME/.bash_private python                                     
                    42

   
EOU

}

private-env(){
   ## only set if not set already .. as is an input used in rum-env 
   [ -z "$ENV_PRIVATE_PATH" ] && export ENV_PRIVATE_PATH=$(private-path)
}
private-name(){ echo .bash_private ; }
private-path(){ echo ${ENV_PRIVATE_PATH:-$(private-path-default)} ; }
private-path-default(){ 
  case ${USER:-nobody} in 
    nobody|www|apache) echo $(dirname $ENV_HOME)/$(private-name) ;;
              default) echo $HOME/$(private-name) ;;
                    *) echo $(dirname $ENV_HOME)/$(private-name) ;;
  esac
}


private-selinux(){
  apache-
  apache-chcon $(apache-private-path)
}

private-sudo(){
  local path=$(private-path)
  [ "$(dirname $path)" == "$HOME" ] && echo -n || echo sudo
}
private-edit(){
  local cmd="$(private-sudo) vi $(private-path) "
  echo $cmd
  eval $cmd
}

private-sync(){
  local msg="=== $FUNCNAME :"

  local cmd
  local ans
  local auser=$(apache- ; apache-user)
  local path=$(USER=$auser private-path)
  local orig=$(private-path default)
  [ "$path" == "$orig" ] && echo $msg skipping as are same path $path && return 0

  cmd="sudo diff $orig $path "
  echo $msg "$cmd" && eval $cmd
  local rc=$?
  [ "$rc" == "0" ] && echo $msg no difference between the files ... skipping && return 0

  cmd="sudo cp $orig $path && sudo chown $auser:$auser $path "  
  read -p "$msg \"$cmd\"  .... enter YES to proceed " ans 
  [ "$ans" != "YES" ] && echo $msg OK skipping && return 1
  eval $cmd

  cmd="sudo chcon -t httpd_sys_content_t $path"
  echo $msg "$cmd" && eval $cmd

  private-ls
}


private-ls(){
   ls -alZ $(USER=apache private-path) $(private-path)
}



private-check-(){
  [ "$1" == "-rw-------"  ] && return 0 || return 1
}



private-hasval-(){ sudo grep $1 $(private-path) > /dev/null ; }
private-set(){     ! private-hasval- $1 && private-append- $* || private-change- $* ;  }
private-append-(){ sudo bash -c "echo local $1=$2 >> $(private-path)   " ;  }
private-change-(){ sudo perl -pi -e "s,local $1=(\S*),local $1=$2, " $(private-path) ; }
private-get(){     sudo perl -n  -e "m,local $1=(\S*), && print \$1" ${2:-$(private-path)} ; }

private-upper(){
   tr "[a-z]" "[A-Z]" 
}


private-export(){
  local var
  for var in $* ; do
    local exp="export $var=$(private-val $var)"
    echo $exp
    eval $exp
  done
}


private-val(){

  local msg="=== $FUNCNAME :" 
  local name=${1:-PRIVATE_VARIABLE}
  local path=${2:-$(private-path)}
  
  [ ! -f "$path" ] && return 1
  !  private-check- $(ls -l $path) ] && echo $msg ABORT inappropriate permissions on path:$path > /dev/stderr && return 1
  [ -f "$path" ] && . $path
 
  # echo $msg determining $name from $path > /dev/stderr
  eval local valu=\$$name
  echo $valu
  return 0
}


private-test(){ 
   [ "$PWD" == "$ENV_HOME" ] && echo $msg ERROR env.pyc will cause problems running from here && return 1
   $FUNCNAME- | python 
}
private-test-(){ cat << EOT

from private import Private
p = Private()
print p('SECRET')

EOT
}

private-ln(){
  python-
  python-ln $(env-home)/private/private
}


private-py-install(){
  cd $(env-home)/private
  $SUDO python setup.py develop
}

private-py-check(){ python -c "from private import Private ; p = Private() ; print p " ;  }
