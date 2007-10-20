
DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE
export DYB=$LOCAL_BASE/dyb

dyb(){      [ -r $DYB_HOME/dyb.bash ]           && . $DYB_HOME/dyb.bash ; } 

dyb-get(){
   mkdir -p $LOCAL_BASE/dyb
   cd $LOCAL_BASE/dyb
   svn export http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst
}


dyb-install-nohup(){
    cd $LOCAL_BASE/dyb
    rm -f nohup.out
    nohup bash -lc "dyb-install $*"
}

dyb-install(){

  local def_arg="all"
  local arg=${1:-$def_arg}
  cd $LOCAL_BASE/dyb
  ./dybinst $arg
}


dyb-env(){
   local dir=$PWD
   cd $DYB
   . sourceme.core
   . core/dybgaudi/DybExamples/ExHelloWorld/cmt/setup.sh
   which python
   cd $dir
}


dyb-sleep(){
  sleep $* && echo "dyb-sleep completed $* " > /tmp/dyb-sleep
}  
  
dyb-smry(){
  cd $LOCAL_BASE/dyb
  tail -f nohup.out
}  
  
dyb-log(){
  cd $LOCAL_BASE/dyb
  tail -f dybinst.log
}  









