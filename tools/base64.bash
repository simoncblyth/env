base64-env(){ echo -n ; } 
base64-vi(){ vi $BASH_SOURCE ; }
base64-usage(){ cat << EOU


EOU
}

base64--(){ 
   #base64-check ; 
   base64-check-file ; 
}

base64-check()
{
    local path=/tmp/env/$FUNCNAME.txt
    mkdir -p $(dirname $path)

    SECRET=secret
    #SECRET=$(<$BASH_SOURCE)
    echo $SECRET > $path

    case $(uname) in 
      Darwin) ENCODED="$(cat $path | base64 -b0)" ; DECODED="$(echo $ENCODED | base64 -D)" ;;
      Linux)  ENCODED="$(cat $path | base64 -w0)" ; DECODED="$(echo $ENCODED | base64 -d)" ;;
    esac

    if [ "$SECRET" == "$DECODED" ]; then MATCH="YES" ; else MATCH="NO" ; fi 


    vv="path SECRET ENCODED DECODED MATCH"
    for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
}

base64-check-file()
{
    local path=/tmp/env/$FUNCNAME
    mkdir -p $(dirname $path)

    SECRET=$BASH_SOURCE
    cat $SECRET > ${path}.secret
    
    if [ "$(uname)" == "Darwin" ]; then 
      
       cat ${path}.secret  | base64 -b0 > ${path}.encoded  
       cat ${path}.encoded | base64 -D  > ${path}.decoded
 
    elif [ "$(uname)" == "Linux" ]; then 

       cat ${path}.secret  | base64 -w0 > ${path}.encoded  
       cat ${path}.encoded | base64 -d  > ${path}.decoded
 
    fi 

    cat ${path}.encoded
    cat ${path}.encoded | wc -l 

    diff -y ${path}.secret ${path}.decoded
    echo $?

}


