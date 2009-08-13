# === func-gen- : cpg/cpg fgp cpg/cpg.bash fgn cpg fgh cpg
cpg-src(){      echo cpg/cpg.bash ; }
cpg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpg-src)} ; }
cpg-vi(){       vi $(cpg-source) ; }
cpg-env(){      elocal- ; }
cpg-usage(){
  cat << EOU
     cpg-src : $(cpg-src)
     cpg-dir : $(cpg-dir)


EOU
}
cpg-name(){ echo cpg14x ; }
cpg-dir(){ echo $(local-base)/env/cpg/$(cpg-name) ; }
cpg-cd(){  cd $(cpg-dir); }
cpg-get(){
   local dir=$(dirname $(cpg-dir)) &&  mkdir -p $dir && cd $dir
   local tgz=$(cpg-name).tar.gz
   [ ! -f "$tgz" ] && curl -L -O "http://downloads.sourceforge.net/project/coppermine/Coppermine/1.4.25%20%28stable%29/$tgz" 
   [ ! -d "$(cpg-name)" ] && tar zxvf $tgz


}

cpg-server(){ echo http://cms01.phys.ntu.edu.tw ; }
cpg-install(){

   cpg-get

   apache-
   apache-ln $(cpg-dir) cpg 
   apache-chown $(cpg-dir) -R

   sudo chcon -R -u system_u -t httpd_sys_content_t $(cpg-dir)

   ## open $(cpg-server)/cpg/install.php

}


