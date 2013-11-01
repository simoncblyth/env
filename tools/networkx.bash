# === func-gen- : tools/networkx fgp tools/networkx.bash fgn networkx fgh tools
networkx-src(){      echo tools/networkx.bash ; }
networkx-source(){   echo ${BASH_SOURCE:-$(env-home)/$(networkx-src)} ; }
networkx-vi(){       vi $(networkx-source) ; }
networkx-env(){      elocal- ; }
networkx-usage(){ cat << EOU

NETWORKX
=========


* http://networkx.github.io/documentation/latest/tutorial/tutorial.html
* http://networkx.lanl.gov/download/networkx/networkx-1.7.tar.gz


1.2 requires py24 or later
1.3,...,1.6,1.7 required py26



::

    simon:networkx-1.2 blyth$ sudo /usr/bin/python setup.py install

EOU
}
#networkx-name(){ echo networkx-1.7 ; }
#networkx-name(){ echo networkx-1.6 ; }
networkx-name(){ echo networkx-1.2 ; }
networkx-dir(){ echo $(local-base)/env/tools/$(networkx-name) ; }
networkx-cd(){  cd $(networkx-dir); }
networkx-mate(){ mate $(networkx-dir) ; }
networkx-get(){
   local dir=$(dirname $(networkx-dir)) &&  mkdir -p $dir && cd $dir


   local nam=$(networkx-name)
   local tgz=$nam.tar.gz
   [ ! -f "$tgz" ] && curl -L -O http://networkx.lanl.gov/download/networkx/$tgz
   [ ! -d "$nam" ] && tar zxvf $tgz 

}
