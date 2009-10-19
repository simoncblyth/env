# === func-gen- : messaging/ahkera fgp messaging/ahkera.bash fgn ahkera fgh messaging
ahkera-src(){      echo messaging/ahkera.bash ; }
ahkera-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ahkera-src)} ; }
ahkera-vi(){       vi $(ahkera-source) ; }
ahkera-env(){      elocal- ; }
ahkera-usage(){
  cat << EOU
     ahkera-src : $(ahkera-src)
     ahkera-dir : $(ahkera-dir)


EOU
}
ahkera-dir(){ echo $(local-base)/env/messaging/ahkera ; }
ahkera-cd(){  cd $(ahkera-dir); }
ahkera-mate(){ mate $(ahkera-dir) ; }
ahkera-get(){
   local dir=$(dirname $(ahkera-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://github.com/t-lo/ahkera.git

}
