
bitrun-usage(){

cat << EOU

    bitrun-url   : $(bitrun-url)
    bitrun-cfg   : $(bitrun-cfg)
    bitrun-path  : $(bitrun-path)

    BITRUN_OPT   : $BITRUN_OPT

    which bitten-slave : $(which bitten-slave)


    bitrun-dumb    :   pure default running 
        
    bitrun-minimal :
    
    bitrun-inplace :
     
       the build-dir is by default created within the work-dir with a name like 
       build_${config}_${build}   setting it to "" as used here  is a convenience for testing
       which MUST go together with "--keep-files" to avoid potentially deleting bits 
       of working copy      
    



EOU
}

bitrun-env(){
  elocal-
  export BITRUN_OPT="--dry-run"
}

bitrun-url(){
   local url
   
   local name=${1:-workflow}
   case $name in
     workflow) url=http://localhost/tracs/$name/builds ;;
          env) url=http://dayabay.phys.ntu.edu.tw/tracs/$name/builds ;;
            *) url=
   esac
   echo $url
}

bitrun-path(){
   case ${1:-workflow} in
     workflow) echo trunk/demo ;;
          env) echo trunk/unittest/demo ;;
            *) echo error-$FUNCNAME ;;
   esac
}

bitrun-cfg(){
    echo $ENV_HOME/bitrun/$LOCAL_NODE.cfg
}

bitrun-fluff(){
    local msg="=== $FUNCNAME: $* "
    local fluff=$WORKFLOW_HOME/demo/fluff.txt
    date >> $fluff
    local cmd="svn ci $fluff -m \"$msg\" "
    echo $cmd
    eval $cmd
}







bitrun-dumb(){
   bitten-slave $(bitrun-url $*)
}

bitrun-minimal(){

   local name=$1
   shift 
   
   ## the less smarts the slave needs the better 

   local msg="=== $FUNCNAME :"
   local cfg=$(bitrun-cfg)
   [ ! -f $cfg ] && echo $msg ERROR no bitten config file $file for LOCAL_NODE $LOCAL_NODE && return 1

   local iwd=$PWD
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   cd $tmp
   
   bitten-slave $* -v --dump-reports -f $cfg  $(bitrun-url $name)
   
   cd $iwd
}


bitrun-cmd-(){

   local name=$1
   shift
     
   local cmd=$(cat << EOC
      bitten-slave -v  -f $(bitrun-cfg) 
         --dump-reports 
         -u blyth -p $NON_SECURE_PASS
           $*
            $(bitrun-url $name)
EOC)

    echo $cmd
}


bitrun-inplace(){

    local name=${1:-$TRAC_INSTANCE}
    shift

    local iwd=$PWD
    local msg="=== $FUNCNAME :"

    [ ! -f $cfg ] && echo $msg ERROR no bitten config file $file for LOCAL_NODE $LOCAL_NODE && return 1

    local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
    cd $tmp
    
    local cmd=$(bitrun-cmd- $name  --work-dir=. --build-dir=  --keep-files) 
    echo $cmd
    eval $cmd
  
    cd $iwd
}









