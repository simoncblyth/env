# === func-gen- : apache/modroot fgp apache/modroot.bash fgn modroot
modroot-src(){      echo apache/modroot.bash ; }
modroot-source(){   echo ${BASH_SOURCE:-$(env-home)/$(modroot-src)} ; }
modroot-vi(){       vi $(modroot-source) ; }
modroot-env(){      elocal- ; }
modroot-usage(){
  cat << EOU
     modroot-src : $(modroot-src)

    $(env-wikiurl)/ModRoot



EOU
}

modroot-url(){ echo ftp://root.cern.ch/root/mod_root2.c ; }
modroot-dir(){ echo $(local-base)/env/apache/modroot ; }
modroot-get(){
   local dir=$(modroot-dir)
   mkdir -p $dir
   cd $dir
   [ ! -f $(basename $(modroot-url)) ] && curl -O $(modroot-url) 
}



