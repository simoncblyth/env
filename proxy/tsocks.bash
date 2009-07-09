# === func-gen- : proxy/tsocks fgp proxy/tsocks.bash fgn tsocks
tsocks-src(){      echo proxy/tsocks.bash ; }
tsocks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tsocks-src)} ; }
tsocks-vi(){       vi $(tsocks-source) ; }
tsocks-env(){      elocal- ; }
tsocks-usage(){
  cat << EOU
     tsocks-src : $(tsocks-src)

EOU
}


tsocks-get(){
   sudo port install tsocks
   sudo port contents tsocks 
}

tsocks-conf-(){ cat << EOC
# $(tsocks-source) $FUNCNAME $(date)
# what is accessed directly ... not thru the proxy 
# where the mask is 0 ... it corresponds to a wildcard in the IP
local = 127.0.0.1/255.255.255.255
local = 208.67.222.222/255.255.255.255

server = 127.0.0.1
server_port = 8080
EOC
}

tsocks-edit(){ sudo vim $(tsocks-conf-path) ;}
tsocks-conf-path(){ echo  /opt/local/etc/tsocks.conf ;  }
tsocks-conf(){
    local msg="=== $FUNCNAME :"
    local path=$(tsocks-conf-path)
    local tmp=/tmp/env/$FUNCNAME/$(basename $path)
    mkdir -p $(dirname $tmp)
    $FUNCNAME- > $tmp

    if [ -f "$path" ] ; then
       diff $path $tmp
    fi
    local ans
    read -p "$msg adopt $tmp for $path ? enter YES to proceed : " ans 
    [ "$ans" != "YES" ] && echo $msg skipping && return 0

    local cmd="sudo cp $tmp $path"
    echo $cmd
    eval $cmd
}

