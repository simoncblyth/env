private-src(){  echo base/private.bash ; }
private-source(){ echo ${BASH_SOURCE:-$(env-home)/$(private-src)} ; }
private-vi(){     vi $(private-source) ; }
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

private-edit(){
  local cmd="sudo vi $(private-path) "
  echo $cmd
  eval $cmd
}

private-path(){
  python -c "from env.base.private import Private ; print Private.path "
  #echo /tmp/env/test.txt
}

private-check-(){
  [ "$1" == "-rw-------"  ] && return 0 || return 1
}



private-hasval-(){ sudo grep $1 $(private-path) > /dev/null ; }
private-set(){ ! private-hasval- $1 && private-append $* || private-change $* ;  }
private-append(){ sudo bash -c "echo local $1=$2 >> $(private-path)   " ;  }
private-change(){ sudo perl -pi -e "s,local $1=(\S*),local $1=$2, " $(private-path) ; }



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

private-selinux(){

  local msg="=== $FUNCNAME :" 
  local path=$(private-path)
  echo $msg $path
  apache-
  apache-chown $path

  local cmd=" sudo chcon -t httpd_sys_content_t $path"
  echo $msg $cmd
  eval $cmd
}

