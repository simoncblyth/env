
private-usage(){

   cat << EOU
   
       Access the values with 
            private-
            private-val PRIVATE_VARIABLE <path>
   
       The path defaults to  $(private-path)     
       The definition file should have variables defined as below :
           local PRIVATE_VARIABLE=42
      
      (this is more secure than exporting them)
      
       returns blank if not defined or if the permissions on the file are 
       not "-rw-------"
       
       returns silently if the file doesnt exist 
   
EOU

}

private-env(){
   echo -n
}

private-path(){
  echo $HOME/.bash_private
}

private-check-(){
  [ "$1" == "-rw-------"  ] && return 0 || return 1
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