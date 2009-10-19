# === func-gen- : messaging/openamq fgp messaging/openamq.bash fgn openamq fgh messaging
openamq-src(){      echo messaging/openamq.bash ; }
openamq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openamq-src)} ; }
openamq-vi(){       vi $(openamq-source) ; }
openamq-env(){      elocal- ; }
openamq-usage(){
  cat << EOU
     openamq-src : $(openamq-src)
     openamq-dir : $(openamq-dir)


EOU
}
openamq-dir(){ echo $(local-base)/env/messaging/OpenAMQ-1.4c0 ; }
openamq-cd(){  cd $(openamq-dir); }
openamq-mate(){ mate $(openamq-dir) ; }
openamq-get(){
   local dir=$(dirname $(openamq-dir)) &&  mkdir -p $dir && cd $dir
   curl -O http://download.imatix.com/openamq/unstable/OpenAMQ-1.4c0.tar.gz && tar zxvf OpenAMQ-1.4c0.tar.gz
}
