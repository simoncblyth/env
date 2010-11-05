# === func-gen- : base/telnet fgp base/telnet.bash fgn telnet fgh base
telnet-src(){      echo base/telnet.bash ; }
telnet-source(){   echo ${BASH_SOURCE:-$(env-home)/$(telnet-src)} ; }
telnet-vi(){       vi $(telnet-source) ; }
telnet-env(){      elocal- ; }
telnet-usage(){
  cat << EOU
     telnet-src : $(telnet-src)
     telnet-dir : $(telnet-dir)


EOU
}
telnet-dir(){ echo $(local-base)/env/base/base-telnet ; }
telnet-cd(){  cd $(telnet-dir); }
telnet-mate(){ mate $(telnet-dir) ; }
telnet-get(){
   local dir=$(dirname $(telnet-dir)) &&  mkdir -p $dir && cd $dir

  telnet-get- | telnet $1 80
}


telnet-get-(){ cat << EOR
GET / http/1.0

EOR
}

