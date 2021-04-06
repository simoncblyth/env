#!/bin/bash 

function main_0(){
    echo $#

    local arg
    declare -a args 
    if [[ $# -eq 0 ]]; then
        args=("get" "check" "install" "red green")
    else
        for arg in "$@" ; do args+=("$arg") ; done
    fi  

    for arg in "${args[@]}" ; do echo $arg ; done
 }


name-get(){ echo $FUNCNAME ; }
name-check(){ echo $FUNCNAME ; }
name-install(){ echo $FUNCNAME ; }

name-red(){ 
   echo $FUNCNAME with args $*  
   local arg
   for arg in "$@" ; do echo $arg ; done
}
name-green(){ echo $FUNCNAME with args $*  ; }
name-blue(){ echo $FUNCNAME with args $*  ; }
name-cyan(){ 
   echo $FUNCNAME with args $*  
   local arg
   for arg in "$@" ; do echo $arg ; done

}


function main_1(){
    declare -a cmds
    [[ $# -eq 0 ]] && cmds=("get" "check" "install" "red green blue") || cmds=("$@") 
 
    local cmd
    for cmd in "${cmds[@]}" 
    do 
        name-$cmd    # sub-cmds feed in as arguments to the function 
    done
}



function main(){

    declare -a cmds
    [[ $# -eq 0 ]] && cmds=("get" "check" "install" "red green blue") || cmds=("$@") 
 
    local cmd
    for cmd in "${cmds[@]}" 
    do 
        name-$cmd    # sub-cmds feed in as arguments to the function 
    done
}



main "$@"


