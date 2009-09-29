# === func-gen- : authkit/authkit fgp authkit/authkit.bash fgn authkit fgh authkit
authkit-src(){      echo authkit/authkit.bash ; }
authkit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(authkit-src)} ; }
authkit-vi(){       vi $(authkit-source) ; }
authkit-env(){      elocal- ; }
authkit-usage(){
  cat << EOU
     authkit-src : $(authkit-src)
     authkit-dir : $(authkit-dir)


EOU
}
authkit-dir(){ echo $(local-base)/env/authkit ; }
authkit-cd(){  cd $(authkit-dir); }
authkit-mate(){ mate $(authkit-dir) ; }


authkit-build(){

   authkit-get


   ! authkit-install && return 1
   ! authkit-selinux && return 1

}



authkit-get(){
   local dir=$(dirname $(authkit-dir)) &&  mkdir -p $dir && cd $dir
   type $FUNCNAME
   hg clone http://bitbucket.org/kumar303/authkit/

}

authkit-diff(){
   local opt=${1:---brief}
   python-
   diff -r $opt $(authkit-dir)/authkit $(python-site)/AuthKit-0.4.4-py$(python-major).egg/authkit | grep -v .pyc 
}

authkit-setup(){
   authkit-cd

   [ -z "$VIRTUAL_ENV" ] && echo $msg ABORT are not inside virtualenv && return 1
   [ "$(which python)" != "$VIRTUAL_ENV/bin/python" ] && echo  $msg ABORT wrong python && return 1

   python setup.py $*
   
}

authkit-selinux(){
   apache-
   apache-chcon $(authkit-dir)
}


authkit-install(){ authkit-setup develop ; }
