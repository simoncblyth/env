# === func-gen- : priv/priv fgp priv/priv.bash fgn priv fgh priv
priv-src(){      echo priv/priv.bash ; }
priv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(priv-src)} ; }
priv-vi(){       vi $(priv-source) ; }
priv-env(){      elocal- ; }
priv-usage(){
  cat << EOU
     priv-src : $(priv-src)
     priv-dir : $(priv-dir)

     priv-build 
     priv-chcon 
         



EOU
}
priv-dir(){ echo $(env-home)/priv ; }
priv-libdir(){ echo $(priv-dir)/lib ; }
priv-cd(){  cd $(priv-dir); }
priv-mate(){ mate $(priv-dir) ; }

priv-chcon(){
  local msg="=== $FUNCNAME :"
  local cmd="sudo chcon -t texrel_shlib_t $(priv-libdir)/libprivate.so"
  echo $msg $cmd
  eval $cmd 
}

priv-build(){
  priv-cd
  make
}

priv-test(){
  priv-cd
  make test
}


