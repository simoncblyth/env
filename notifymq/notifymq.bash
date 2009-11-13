# === func-gen- : notifymq/notifymq fgp notifymq/notifymq.bash fgn notifymq fgh notifymq
notifymq-src(){      echo notifymq/notifymq.bash ; }
notifymq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(notifymq-src)} ; }
notifymq-vi(){       vi $(notifymq-source) ; }
notifymq-env(){      elocal- ; }
notifymq-usage(){
  cat << EOU
     notifymq-src : $(notifymq-src)
     notifymq-dir : $(notifymq-dir)


EOU
}

notifymq-preq(){

   rabbitmq-c-get


}



notifymq-dir(){ echo $(local-base)/env/notifymq/notifymq-notifymq ; }
notifymq-cd(){  cd $(notifymq-dir); }
notifymq-mate(){ mate $(notifymq-dir) ; }
notifymq-get(){
   local dir=$(dirname $(notifymq-dir)) &&  mkdir -p $dir && cd $dir

}




