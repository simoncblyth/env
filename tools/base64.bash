base64-env(){ echo -n ; } 
base64-vi(){ vi $BASH_SOURCE ; }
base64-usage(){ cat << EOU


EOU
}

base64--(){ base64-check ; }

base64-check()
{
    local path=/tmp/env/$FUNCNAME.txt
    mkdir -p $(dirname $path)

    SECRET=secret
    echo $SECRET > $path
    case $(uname) in 
      Darwin) ENCODED="$(cat $path | base64 -b0)" ; DECODED="$(echo $ENCODED | base64 -D)" ;;
      Linux)  ENCODED="$(cat $path | base64 -w0)" ; DECODED="$(echo $ENCODED | base64 -d)" ;;
    esac

    if [ "$SECRET" == "$DECODED" ]; then MATCH="YES" ; else MATCH="NO" ; fi 


    vv="path SECRET ENCODED DECODED MATCH"
    for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
}



