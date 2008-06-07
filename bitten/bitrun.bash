
bitrun-usage(){

cat << EOU

    which bitten-slave : $(which bitten-slave)

EOU
}


bitrun-env(){
  elocal-
  export BITRUN_OPT="--dry-run"
}

bitrun-url(){
   local url
   case ${1:-workflow} in
     workflow) url=http://localhost/tracs/$1/builds
          env) url=http://dayabay.phys.ntu.edu.tw/tracs/$1/builds
            *) url=
   esac
   echo $url
}

bitrun-default(){
   bitten-slave $(bitrun-url $*)
}


bitrun-minimal(){

   local name=$1
   shift 
   
   ## the less smarts the slave needs the better 

   local msg="=== $FUNCNAME :"
   local cfg=$ENV_HOME/bitten/$LOCAL_NODE.cfg

   [ ! -f $cfg ] && echo $msg ERROR no bitten config file $file for LOCAL_NODE $LOCAL_NODE && return 1

   local iwd=$PWD
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   cd $tmp
   
   bitten-slave $* -v --dump-reports -f $cfg  $(bitrun-url $name)
   
   cd $iwd
}


bitrun-fluff(){

    local msg="=== $FUNCNAME: $* "
    local fluff=$ENV_HOME/unittest/demo/fluff.txt
    date >> $fluff
    local cmd="svn ci $fluff -m \"$msg\" "
    echo $cmd
    eval $cmd
}

bitrun-cfg(){
    echo $ENV_HOME/bitten/$LOCAL_NODE.cfg
}


bitrun-inplace(){

    local iwd=$PWD
    local msg="=== $FUNCNAME :"
    local cfg=$(bitrun-cfg)
    [ ! -f $cfg ] && echo $msg ERROR no bitten config file $file for LOCAL_NODE $LOCAL_NODE && return 1

    local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
    cd $tmp
    
    local cmd=$(cat << EOC
     bitten-slave -v $arg  -f $cfg 
         --dump-reports 
          --work-dir=.
         --build-dir=
         --keep-files 
            $(bitrun-url $name)
EOC)
    echo $cmd
    eval $cmd

   #
   #  the build-dir is by default created within the work-dir with a name like 
   #   build_${config}_${build}   setting it to "" is a convenience for testing
   #  which MUST go together with "--keep-files" to avoid potentially deleting bits of working copy  
   #


    cd $iwd
}









