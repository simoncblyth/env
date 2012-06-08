# === func-gen- : tools/daemonwatch fgp tools/daemonwatch.bash fgn daemonwatch fgh tools
daemonwatch-src(){      echo tools/daemonwatch.bash ; }
daemonwatch-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daemonwatch-src)} ; }
daemonwatch-vi(){       vi $(daemonwatch-source) ; }
daemonwatch-env(){      elocal- ; }
daemonwatch-usage(){ cat << EOU

Daemonwatch
============


Installs
---------

G
   





EOU
}
daemonwatch-dir(){ echo $(local-base)/env/tools/daemonwatch ; }
daemonwatch-cd(){  cd $(daemonwatch-dir); }
daemonwatch-mate(){ mate $(daemonwatch-dir) ; }
daemonwatch-get(){
   local dir=$(dirname $(daemonwatch-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://github.com/sebastien/daemonwatch.git
}

daemonwatch-install(){
   daemonwatch-cd
   sudo python setup.py install
}

