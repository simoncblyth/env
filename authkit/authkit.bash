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
authkit-dir(){ echo $(local-base)/env/$(authkit-name) ; }
authkit-name(){ echo AuthKitPy24 ; }
authkit-rel(){ echo AuthKit/trunk ; }
authkit-cd(){  cd $(authkit-dir)/$*; }
authkit-mate(){ mate $(authkit-dir) ; }


authkit-build(){

   authkit-get
   [ ! $? -eq 0 ] && return 1
   authkit-install 
   [ ! $? -eq 0 ] && return 1
   authkit-selinux 
   [ ! $? -eq 0 ] && return 1

   return 0
}


authkit-get(){
   local dir=$(dirname $(authkit-dir)) &&  mkdir -p $dir && cd $dir
   type $FUNCNAME
   if [ -d "$(authkit-name)" ] ; then
      ( cd $(authkit-name) ; hg pull )
   else
      hg clone http://belle7.nuu.edu.tw/hg/$(authkit-name)
   fi  
}

#authkit-diff(){
#   local opt=${1:---brief}
#   python-
#   diff -r $opt $(authkit-dir)/authkit $(python-site)/AuthKit-0.4.4-py$(python-major).egg/authkit | grep -v .pyc 
#}

authkit-setup(){
   authkit-cd $(authkit-rel)

   [ -z "$VIRTUAL_ENV" ] && echo $msg ABORT are not inside virtualenv && return 1
   [ "$(which python)" != "$VIRTUAL_ENV/bin/python" ] && echo  $msg ABORT wrong python && return 1

   python setup.py $*
   
}

authkit-selinux(){
   apache-
   apache-chcon $(authkit-dir)
}


authkit-install(){ authkit-setup develop ; }






