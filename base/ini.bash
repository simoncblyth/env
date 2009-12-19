# === func-gen- : base/ini fgp base/ini.bash fgn ini fgh base
ini-src(){      echo base/ini.bash ; }
ini-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ini-src)} ; }
ini-vi(){       vi $(ini-source) ; }
ini-env(){      elocal- ; }
ini-usage(){
  cat << EOU
     ini-src : $(ini-src)
     ini-dir : $(ini-dir)


EOU
}
ini-dir(){ echo $(local-base)/env/base/base-ini ; }
ini-cd(){  cd $(ini-dir); }
ini-mate(){ mate $(ini-dir) ; }
ini-get(){
   local dir=$(dirname $(ini-dir)) &&  mkdir -p $dir && cd $dir

}

ini-flavor(){ echo ${INI_FLAVOR:-ini} ; }
ini-py(){ echo $(env-home)/base/$(ini-flavor).py ; } 
ini-triplet-edit()
{
    local msg="=== $FUNCNAME :";
    local path=$1 ;
    shift;
    local tmp=/tmp/$USER/env/$FUNCNAME && mkdir -p $tmp;
    local tpath=$tmp/$(basename $path);
    local cmd="cp $path $tpath ";
    eval $cmd;
    INI_TRIPLET_DELIM="|" python $(ini-py) $tpath $*;
    local dmd="diff $path $tpath";
    echo $msg $dmd;
    eval $dmd;
    [ "$?" == "0" ] && echo $msg no differences ... skipping && return 0;
    if [ -n "$INI_CONFIRM" ]; then
        local ans;
        read -p "$msg enter YES to confirm this change " ans;
        [ "$ans" != "YES" ] && echo $msg skipped && return 1;
    fi;
    $SUDO cp $tpath $path;

    #[ "$user" != "$USER" ] && $SUDO chown $user:$user $path
}






