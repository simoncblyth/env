# === func-gen- : nuwa/dybinst fgp nuwa/dybinst.bash fgn dybinst fgh nuwa
dybinst-src(){      echo nuwa/dybinst.bash ; }
dybinst-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dybinst-src)} ; }
dybinst-vi(){       vi $(dybinst-source) ; }
dybinst-env(){      elocal- ; }
dybinst-usage(){ cat << EOU





EOU
}
dybinst-dir(){ echo $(local-base)/env/dyb ; }
dybinst-cd(){  cd $(dybinst-dir); }
dybinst-mate(){ mate $(dybinst-dir) ; }
dybinst-url(){     echo http://dayabay.ihep.ac.cn/svn/dybsvn/installation/trunk/dybinst/dybinst ; }
dybinst-get(){
   local dir=$(dybinst-dir) &&  mkdir -p $dir && cd $dir
   [ ! -f dybinst ] && svn export $(dybinst-url)    
}
dybinst-all(){
    dybinst-cd
    ./dybinst trunk all  
}


