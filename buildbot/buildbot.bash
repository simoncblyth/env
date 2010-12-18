# === func-gen- : buildbot/buildbot fgp buildbot/buildbot.bash fgn buildbot fgh buildbot
buildbot-src(){      echo buildbot/buildbot.bash ; }
buildbot-source(){   echo ${BASH_SOURCE:-$(env-home)/$(buildbot-src)} ; }
buildbot-vi(){       vi $(buildbot-source) ; }
buildbot-env(){      elocal- ; }
buildbot-usage(){
  cat << EOU
     buildbot-src : $(buildbot-src)
     buildbot-dir : $(buildbot-dir)

     http://trac.buildbot.net/


EOU
}
buildbot-dir(){ echo $(local-base)/env/buildbot/buildbot-buildbot ; }
buildbot-cd(){  cd $(buildbot-dir); }
buildbot-mate(){ mate $(buildbot-dir) ; }
buildbot-get(){
   local dir=$(dirname $(buildbot-dir)) &&  mkdir -p $dir && cd $dir

}
