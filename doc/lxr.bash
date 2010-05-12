# === func-gen- : doc/lxr fgp doc/lxr.bash fgn lxr fgh doc
lxr-src(){      echo doc/lxr.bash ; }
lxr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lxr-src)} ; }
lxr-vi(){       vi $(lxr-source) ; }
lxr-env(){      elocal- ; }
lxr-usage(){
  cat << EOU
     lxr-src : $(lxr-src)
     lxr-dir : $(lxr-dir)

     http://sourceforge.net/projects/lxr/

EOU
}

lxr-name(){ echo lxr-0.9.8 ; }
lxr-url(){  echo http://downloads.sourceforge.net/project/lxr/stable/$(lxr-name)/$(lxr-name).tgz?use_mirror=ncu ;  }
lxr-dir(){ echo $(local-base)/env/doc/$(lxr-name) ; }
lxr-cd(){  cd $(lxr-dir); }
lxr-mate(){ mate $(lxr-dir) ; }
lxr-get(){
   local dir=$(dirname $(lxr-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -f "$(lxr-name).tgz" ] && curl -L -o $(lxr-name).tgz $(lxr-url) 
   [ ! -d "$(lxr-name)"     ] && tar zxvf $(lxr-name).tgz
}
