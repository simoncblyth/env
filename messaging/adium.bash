# === func-gen- : messaging/adium fgp messaging/adium.bash fgn adium fgh messaging
adium-src(){      echo messaging/adium.bash ; }
adium-source(){   echo ${BASH_SOURCE:-$(env-home)/$(adium-src)} ; }
adium-vi(){       vi $(adium-source) ; }
adium-env(){      elocal- ; }
adium-usage(){
  cat << EOU
     adium-src : $(adium-src)
     adium-dir : $(adium-dir)


EOU
}
adium-dir(){ echo "~/Library/Application Support/Adium 2.0/Users/Default/libpurple" ; }
adium-cd(){  cd $(adium-dir); }
adium-mate(){ mate $(adium-dir) ; }
adium-get(){
   local dir=$(dirname $(adium-dir)) &&  mkdir -p $dir && cd $dir

}
