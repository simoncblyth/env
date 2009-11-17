# === func-gen- : messaging/alice fgp messaging/alice.bash fgn alice fgh messaging
alice-src(){      echo messaging/alice.bash ; }
alice-source(){   echo ${BASH_SOURCE:-$(env-home)/$(alice-src)} ; }
alice-vi(){       vi $(alice-source) ; }
alice-env(){      elocal- ; }
alice-usage(){
  cat << EOU
     alice-src : $(alice-src)
     alice-dir : $(alice-dir)


EOU
}
alice-dir(){ echo $(local-base)/env/messaging/alice ; }
alice-cd(){  cd $(alice-dir); }
alice-mate(){ mate $(alice-dir) ; }
alice-get(){
   local dir=$(dirname $(alice-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/auser/alice.git

}
