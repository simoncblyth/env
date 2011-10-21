# === func-gen- : base/tcpdump fgp base/tcpdump.bash fgn tcpdump fgh base
tcpdump-src(){      echo base/tcpdump.bash ; }
tcpdump-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tcpdump-src)} ; }
tcpdump-vi(){       vi $(tcpdump-source) ; }
tcpdump-env(){      elocal- ; }
tcpdump-usage(){
  cat << EOU
     tcpdump-src : $(tcpdump-src)
     tcpdump-dir : $(tcpdump-dir)

     Packet analysis

     http://www.danielmiessler.com/study/tcpdump/

EOU
}
tcpdump-dir(){ echo $(local-base)/env/base/base-tcpdump ; }
tcpdump-cd(){  cd $(tcpdump-dir); }
tcpdump-mate(){ mate $(tcpdump-dir) ; }
tcpdump-get(){
   local dir=$(dirname $(tcpdump-dir)) &&  mkdir -p $dir && cd $dir

}
