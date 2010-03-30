# === func-gen- : root/rootana fgp root/rootana.bash fgn rootana fgh root
rootana-src(){      echo root/rootana.bash ; }
rootana-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rootana-src)} ; }
rootana-vi(){       vi $(rootana-source) ; }
rootana-env(){      elocal- ; }
rootana-usage(){
  cat << EOU
     rootana-src : $(rootana-src)
     rootana-dir : $(rootana-dir)


EOU
}
rootana-dir(){ echo $(local-base)/env/root/rootana ; }
rootana-cd(){  cd $(rootana-dir); }
rootana-mate(){ mate $(rootana-dir) ; }
rootana-get(){
   local dir=$(dirname $(rootana-dir)) &&  mkdir -p $dir && cd $dir
   svn checkout $(rootana-url) rootana
}

rootana-url(){ echo svn://ladd00.triumf.ca/rootana/trunk  ; }


