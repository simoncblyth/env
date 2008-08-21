
private-usage(){

   cat << EOU
  
       Private variables should be defined in $(private-path) in the form
           local PRIVATE_VARIABLE=42
       (this is preferable to exporting them for all and sundy to see)
      
      
       Access the value with 
            private-
            private-val PRIVATE_VARIABLE 
       returns blank if not defined 
   
   
EOU


}

private-env(){
   elocal-
}

private-path(){
  echo $HOME/.bash_private
}

private-check-(){
  [ "$1" == "-rw-------"  ] && return 0 || return 1
}

private-access(){
  local path=${1:-$(private-path)}
  !  private-check- $(ls -l $path) ] && return 1
  [ -f "$path" ] && . $path
  return 0
}

private-val(){

  local name=$1
  local path=${2:-$(private-path)}
  
  local msg="=== $FUNCNAME :"
  ! private-access $path && echo $msg ABORT inappropriate permissions on path:$path
  eval valu=\$$name
  echo $valu
}