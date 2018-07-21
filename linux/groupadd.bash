# === func-gen- : linux/groupadd fgp linux/groupadd.bash fgn groupadd fgh linux
groupadd-src(){      echo linux/groupadd.bash ; }
groupadd-source(){   echo ${BASH_SOURCE:-$(env-home)/$(groupadd-src)} ; }
groupadd-vi(){       vi $(groupadd-source) ; }
groupadd-env(){      elocal- ; }
groupadd-usage(){ cat << EOU

groupadd : setup shared opticks group 
==========================================

::

   sudo groupadd opticks
   sudo usermod -a -G opticks blyth
   sudo usermod -a -G opticks simon

* https://www.tecmint.com/create-a-shared-directory-in-linux/

Actually its a bit unclear what to share ? Its convenient to 
have separate installs, but thats heavy ?




EOU
}
groupadd-dir(){ echo $(local-base)/env/linux/linux-groupadd ; }
groupadd-cd(){  cd $(groupadd-dir); }
groupadd-mate(){ mate $(groupadd-dir) ; }
groupadd-get(){
   local dir=$(dirname $(groupadd-dir)) &&  mkdir -p $dir && cd $dir

}
