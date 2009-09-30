# === func-gen- : authkit/authkitdev fgp authkit/authkitdev.bash fgn authkitdev fgh authkit
authkitdev-src(){      echo authkit/authkitdev.bash ; }
authkitdev-source(){   echo ${BASH_SOURCE:-$(env-home)/$(authkitdev-src)} ; }
authkitdev-vi(){       vi $(authkitdev-source) ; }
authkitdev-env(){      elocal- ; }
authkitdev-usage(){
  cat << EOU
     authkitdev-src : $(authkitdev-src)
     authkitdev-dir : $(authkitdev-dir)


EOU
}
authkitdev-dir(){ echo $(local-base)/env/authkitdev/$(authkitdev-orig) ; }
authkitdev-cd(){  cd $(authkitdev-dir)/$*; }
authkitdev-mate(){ mate $(authkitdev-dir) ; }
authkitdev-urlbase(){ echo http://belle7.nuu.edu.tw/hg ; }

authkitdev-rel(){  echo AuthKit/trunk ; }
authkitdev-orig(){ echo AuthKitPy24 ; }
authkitdev-fork(){ echo AuthKit_kumar303 ; }
authkitdev-get(){
   local dir=$(dirname (authkitdev-dir)) &&  mkdir -p $dir && cd $dir

   hg clone $(authkitdev-urlbase)/$(authkitdev-orig)
   ## hg clone $(authkitdev-urlbase)/$(authkitdev-fork)     just for reference
}

authkitdev-diff(){
   authkitdev-cd
   

   diff -r --brief ../$(authkitdev-orig)/$(authkitdev-rel) ../$(authkitdev-fork) | grep -v .hg

}

authkitdev-setup(){
  authkitdev-cd $(authkitdev-rel)
  python setup.py $* 
}

authkitdev-build(){
   authkitdev-get
   authkitdev-install
   authkitdev-selinux
}

authkitdev-selinux(){
  apache-
  apache-chcon $(authkitdev-dir)/$(authkitdev-rel)
}


authkitdev-install(){ authkitdev-setup develop ; }

