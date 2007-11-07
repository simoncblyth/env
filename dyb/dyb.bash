
DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE
export DYB=$LOCAL_BASE/dyb

dyb(){      [ -r $DYB_HOME/dyb.bash ]           && . $DYB_HOME/dyb.bash ; } 

dyb-get(){
   local branch=${1:-trunk}
   mkdir -p $LOCAL_BASE/dyb
   cd $LOCAL_BASE/dyb
   
   if [ "X$branch" == "Xtrunk" ]; then 
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst
   else
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/branches/inst-NuWa-$branch/dybinst/dybinst
   fi
   echo === dyb-get branch $branch url $url == see https://wiki.bnl.gov/dayabay/index.php?title=Offline_Release_rozz-0.0.4 ==
   svn export $url
    
}


dyb-linklog(){

  cd $LOCAL_BASE/dyb
  rm -f dybinst.log
  local log=$(ls -tr dybinst-*.log|tail -1)
  local cmd="ln -s $log dybinst.log"
  echo === dyb-linklog $cmd ===
  eval $cmd 

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
  dyb-linklog
  tail -f dybinst.log
}  









