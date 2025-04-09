base64-env(){ echo -n ; } 
base64-vi(){ vi $BASH_SOURCE ; }
base64-usage(){ cat << EOU


Note that the old Darwin base64 adds newline at the end of the encoded string
causing wc -l to return 1::

    epsilon:env blyth$ which base64
    /usr/bin/base64
    epsilon:env blyth$ ls -alst /usr/bin/base64
    16 -rwxr-xr-x  1 root  wheel  23248 Jan 19  2018 /usr/bin/base64
    epsilon:env blyth$ 


    A[blyth@localhost env]$ which base64
    /bin/base64
    A[blyth@localhost env]$ ls -alst /bin/base64
    36 -rwxr-xr-x. 1 root root 36560 Oct  3 05:44 /bin/base64




    epsilon:env blyth$ cat /tmp/env/base64-check-file.encoded | wc 
           1       1    1777

    A[blyth@localhost env]$ cat /tmp/env/base64-check-file.encoded | wc 
           0       1    1776



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
      
       cat ${path}.secret  | base64 -b0 | tr -d '\n' > ${path}.encoded  
       cat ${path}.encoded | base64 -D  > ${path}.decoded
 
    elif [ "$(uname)" == "Linux" ]; then 

       cat ${path}.secret  | base64 -w0 > ${path}.encoded  
       cat ${path}.encoded | base64 -d  > ${path}.decoded
 
    fi 

    echo ==== cat ${path}.encoded
    cat ${path}.encoded

    echo ==== cat ${path}.encoded \| wc
    cat ${path}.encoded | wc  

    echo ==== diff -y ${path}.secret ${path}.decoded
    diff -y ${path}.secret ${path}.decoded
    echo $?

}


base64-encode()
{
    local path=$1
    case $(uname) in 
       Darwin) cat $path  | base64 -b0 | tr -d '\n'  ;;
       Linux)  cat $path | base64 -w0 ;;
    esac 
}

base64-decode()
{
    local path=$1
    case $(uname) in 
       Darwin) cat $path | base64 -D  ;;
       Linux)  cat $path | base64 -d ;;
    esac 
}




