# === func-gen- : base/redhat fgp base/redhat.bash fgn redhat fgh base
redhat-src(){      echo base/redhat.bash ; }
redhat-source(){   echo ${BASH_SOURCE:-$(env-home)/$(redhat-src)} ; }
redhat-vi(){       vi $(redhat-source) ; }
redhat-env(){      elocal- ; }
redhat-usage(){
  cat << EOU
     redhat-src : $(redhat-src)
     redhat-dir : $(redhat-dir)

     redhat-epel
         Configure yum to have access to EPEL,  Extra Packages for Enterprise Linux  
         http://fedoraproject.org/wiki/EPEL/FAQ#howtouse

EOU
}
redhat-dir(){ echo $(local-base)/env/base/base-redhat ; }
redhat-cd(){  cd $(redhat-dir); }
redhat-mate(){ mate $(redhat-dir) ; }
redhat-get(){
   local dir=$(dirname $(redhat-dir)) &&  mkdir -p $dir && cd $dir

}

redhat-vers(){
  cat /etc/redhat-release
}


redhat-epel(){
   sudo rpm -Uvh http://download.fedora.redhat.com/pub/epel/4/i386/epel-release-4-9.noarch.rpm

}



