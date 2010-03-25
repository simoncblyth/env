# === func-gen- : nuwa/daily fgp nuwa/daily.bash fgn daily fgh nuwa
daily-src(){      echo nuwa/daily.bash ; }
daily-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daily-src)} ; }
daily-vi(){       vim $(daily-source) ; }
daily-env(){      elocal- ; }
daily-usage(){
  cat << EOU
     daily-src : $(daily-src)
     daily-dir : $(daily-dir)


EOU
}
daily-dir(){
  case $(hostname) in 
     lxslc??.ihep.ac.cn) echo /home/dyb/dybsw/NuWa/daily ;; 
        farm1.dyb.local) echo /home/dyb/dybsw/NuWa/daily ;; 
                      *) echo /tmp/env/$FUNCNAME         ;;
  esac 
}
daily-cd(){  cd $(daily-dir); }
daily-tcd(){  cd $(daily-todaydir); }
daily-mate(){ mate $(daily-dir) ; }
daily-get(){
   local dir=$(dirname $(daily-dir)) &&  mkdir -p $dir && cd $dir
}

daily-creds(){ echo "dayabay:$(<~/.dybpass)" ; }
daily-url(){   echo "http://dayabay.ihep.ac.cn/tracs/dybsvn/daily/dybinst?format=txt" ; }
daily-rev-(){   curl --fail -s -u "$(daily-creds)" "$(daily-url)" ; }
daily-rev(){
  local msg="=== $FUNCNAME :"  
  local rev
  local rc
  rev=$(daily-rev-)
  rc=$?
  ## cannot combine prior 4 lines to 2 and capture both output and rc 
  case $rc in
     0) echo $msg last revision is $rev ;;
     *) echo $msg FAILED with to access last revision ... curl gave non-zero rc $rc  ... bad credentials OR bad server config  ;;
  esac
}

daily-builddir(){ echo NuWa-$1 ; }
daily-daydir(){   echo NuWa-$(date +"%Y%m%d") ; }
daily-todaydir(){ echo $(daily-dir)/$(daily-daydir) ; }
daily-build(){
   local msg="=== $FUNCNAME :"
   local rev
   rev=$(daily-rev-)
   rc=$?
   [ "$rc" != "0" ] && echo $msg ABORT failed to obtain revision rc $rc && return 1    

   local ddir=$(daily-dir)
   mkdir -p $ddir && cd $ddir
   
   daily-build- $rev
   rc=$?
   [ "$rc" != "0" ] && echo $msg ABORT && return $rc     

   daily-cd
   ln -sf $(daily-builddir $rev) $(daily-daydir)
}

daily-dybinst-url(){     echo http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst ; }
daily-build-(){
  local msg="=== $FUNCNAME :"
  local rev=$1
  local bdir=$(daily-builddir $rev)
  [ -z "$rev" ]  && echo $msg revision argument required && return 1 
  # [ -d "$bdir" ] && echo $msg builddir $bdir exists already && return 0 

  echo $msg proceeding with build $bdir ...
  local iwd=$PWD
  mkdir -p $bdir && cd $bdir

  [ -f "dybinst" ] && echo $msg dybinst already exported || svn export $(daily-dybinst-url)

  local cmd
  cmd="./dybinst -z $rev -e ../../external trunk checkout"
  echo $msg $cmd
  eval $cmd
 
  cmd="./dybinst -z $rev -e ../../external -c trunk projects"
  echo $msg $cmd
  eval $cmd


  cd $iwd
}



