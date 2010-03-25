# === func-gen- : nuwa/daily fgp nuwa/daily.bash fgn daily fgh nuwa
daily-src(){      echo nuwa/daily.bash ; }
daily-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daily-src)} ; }
daily-vi(){       vi $(daily-source) ; }
daily-env(){      elocal- ; }
daily-usage(){
  cat << EOU
     daily-src : $(daily-src)
     daily-dir : $(daily-dir)


EOU
}
daily-dir(){ echo $(local-base)/env/nuwa/nuwa-daily ; }
daily-cd(){  cd $(daily-dir); }
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







