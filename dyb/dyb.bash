

DYB_BASE=$ENV_BASE/dyb
export DYB_HOME=$HOME/$DYB_BASE



dyb-version(){

  # note, sensitivity to preset DYB_VERSION ... overrides the below setting
  export DYB_VERSION_P=trunk
  #export DYB_VERSION_P=0.0.4  
  
 if [ "X$DYB_VERSION" == "X" ]; then
   vname=DYB_VERSION_$NODE_TAG
   eval DYB_VERSION=\$$vname
 else
   echo WARNING honouring a preset DYB_VERSION $DYB_VERSION     
 fi
 
 
 
 export DYB_OPTION=""
 #export DYB_OPTION="_dbg"
 
 export DYB_VERSION
 export DYB_FOLDER=$LOCAL_BASE/dyb
 export DYB=$DYB_FOLDER/$DYB_VERSION$DYB_OPTION 

 export DYB_RELEASE=NuWa-$DYB_VERSION

 ## next time distinguish the options (particulary debug on or off status) via the folder name also 


}

dyb-version


dyb(){      [ -r $DYB_HOME/dyb.bash ]           && . $DYB_HOME/dyb.bash ; } 

dyb-get(){
   
   ## get the branch from the operating directory 
   mkdir -p $DYB
   cd $DYB
   local branch=$(basename $PWD)
   
   if [ "X$branch" == "Xtrunk" ]; then 
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst
   else
     url=http://dayabay.ihep.ac.cn/svn/dybsvn/installation/branches/inst-NuWa-$branch/dybinst/dybinst
   fi
   echo === dyb-get branch $branch url $url == see https://wiki.bnl.gov/dayabay/index.php?title=Offline_Release_rozz-0.0.4 ==
   svn export $url
    
}


dyb-check(){

  cd $DYB
  local version=$(basename $PWD)
  
  if [ "$version" == "$DYB_VERSION" ]; then
     echo === dyb-check consistent versions $version ==
  else
     echo === dyb-check INCONSITENT VERSIONS ... DYB_VERSION $DYB_VERSION version $version DYB $DYW ===
  fi
}





dyb-linklog(){

  cd $DYB
  rm -f dybinst.log
  local log=$(ls -tr dybinst-*.log|tail -1)
  local cmd="ln -s $log dybinst.log"
  echo === dyb-linklog $cmd ===
  eval $cmd 

}

dyb-install-nohup(){
    cd $DYB
    rm -f nohup.out
    nohup bash -lc "dyb-install $*"
}


dyb-install-screen(){
   cd $DYB
   screen bash -lc "dyb-install $*"
}


dyb-install(){
  ## "all" if no argument given, otherwise propagate  
  cd $DYB
  ./dybinst ${*:-all}
}


dyb-source(){
  local pwd=$PWD
  cd $DYB
  . sourceme.NuWa
  cd $pwd
}


dyb-examples-run(){

  local example=${1:-ExHelloWorld}
  dyb-examples-setup $example
  
  case "$example" in
     ExHelloWorld) invoker=HelloWorldInPython ;;
                *) invoker=NONE ;;
  esac
  
  if [ "X$invoker" == "XNONE" ]; then
     echo === dyb-examples-run  example $example has no corresponding invoker  === 
  else
     which python
     local dir=$PWD 
     cd $DYB
     local cmd="python $DYB_RELEASE/dybgaudi/DybExamples/$example/share/$invoker.py"
     echo $cmd
     eval $cmd
     cd $dir 
  fi  

}

dyb-examples-setup(){

   local example=${1:-ExHelloWorld}
   local dir=$PWD
   
   cd $DYB
   . sourceme.$DYB_RELEASE
   . $DYB_RELEASE/dybgaudi/DybExamples/$example/cmt/setup.sh
   
   which python
   cd $dir
}


dyb-sleep(){
  sleep $* && echo "dyb-sleep completed $* " > /tmp/dyb-sleep
}  
  
dyb-smry(){
  cd $DYB
  tail -f nohup.out
}  
  
dyb-log(){
  cd $DYB
  dyb-linklog
  tail -f dybinst.log
}  









