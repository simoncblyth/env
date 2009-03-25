log-src(){ echo log/log.bash ; }
log-source(){  echo ${BASH_SOURCE:-$(env-home)/$(log-src)} ; }
log-vi(){      vi $(log-source) ; }
log-env(){  elocal- ; }
log-usage(){
   cat << EOU

      log-init  <context-name>

      log-tail

      log-cat 
          
      log--
          echo hello | log--


EOU
}


log-dir(){
    local name=${1:-$FUNCNAME}
    local logd=/tmp/env/logs/$name && mkdir -p $logd
    echo $logd
}
log-name(){ echo last.log ; }
log-path(){ echo $(log-dir $*)/$(log-name) ; }
log-stamp(){ date +%Y%m%d-%H%M%S ; }
log-init(){
    local msg="=== $FUNCNAME :"
    local iwd=$PWD
    local name=${1:-$FUNCNAME}
    local logd=$(log-dir $*)
    cd $logd
    local logn=$(log-stamp).log
    cat << EOI
$msg initializing a log $logn at $logd 
  ... write to the log via symbolic link 
               \$(log-path \$name) : $(log-path $name)
  ... or by piping to 
           some command | env-log \$name

  follow the stdout from another term with 
          log-tail $name
          log-cat  $name

EOI
    ln -sf $logn $(log-name)
    echo $msg $* logd $logd $(date) > $logn   # prime/truncate
    cd $iwd
}

log--(){
   local msg="=== $FUNCNAME :"
   local name=$1
   local path=$(log-path $name)
   shift
   local smry="$msg $* logging to $path $(date)"
   echo $smry 
   echo $smry >> $path
   cat      - >> $path
}

log-cat(){   cat $(log-path $*) ; }
log-tail(){  tail -f $(log-path $*) ; } 



