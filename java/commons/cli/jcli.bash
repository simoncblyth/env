# === func-gen- : java/commons/cli/jcli fgp java/commons/cli/jcli.bash fgn jcli fgh java/commons/cli
jcli-src(){      echo java/commons/cli/jcli.bash ; }
jcli-source(){   echo ${BASH_SOURCE:-$(env-home)/$(jcli-src)} ; }
jcli-vi(){       vi $(jcli-source) ; }
jcli-env(){      elocal- ; }
jcli-usage(){ cat << EOU

JAVA APACHE COMMONS CLI
========================

* http://commons.apache.org/proper/commons-cli/
* http://commons.apache.org/proper/commons-cli/introduction.html

Building from source required Maven 2. Checking macports show this to depend on kaffe
(a complete virtual environment that will take ages to install so go with the binary).

MANUAL HASH CHECK
------------------
::

    simon:cli blyth$ md5 commons-cli-1.2-bin.tar.gz
    MD5 (commons-cli-1.2-bin.tar.gz) = a05956c9ac8bacdc2b8d07fb2cb331ce
    simon:cli blyth$ echo $(curl -s http://www.apache.org/dist/commons/cli/binaries/commons-cli-1.2-bin.tar.gz.md5)
    a05956c9ac8bacdc2b8d07fb2cb331ce

EOU
}
jcli-dir(){ echo $(local-base)/env/java/commons/cli/$(jcli-leaf) ; }
jcli-cd(){  cd $(jcli-dir); }
jcli-mate(){ mate $(jcli-dir) ; }
jcli-mode(){ echo bin ; }
jcli-name(){ echo commons-cli-1.2 ; }
jcli-leaf(){  
   case $(jcli-mode) in 
     src) echo $(jcli-name)-src ;; 
     bin) echo $(jcli-name) ;;
   esac
}
jcli-url(){
   case $(jcli-mode) in 
      src) echo http://ftp.mirror.tw/pub/apache//commons/cli/source/$(jcli-name)-$(jcli-mode).tar.gz ;;
      bin) echo http://ftp.mirror.tw/pub/apache//commons/cli/binaries/$(jcli-name)-$(jcli-mode).tar.gz ;;
   esac
}
jcli-jar(){ echo $(jcli-dir)/$(jcli-name).jar ; }
jcli-get(){
   local dir=$(dirname $(jcli-dir)) &&  mkdir -p $dir && cd $dir

   local nam=$(jcli-name)
   local url=$(jcli-url)
   local tgz=$(basename $url)

   local nam=${tgz/.tar.gz}
   local leaf=$(jcli-leaf)

   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$leaf" ] && tar zxvf $tgz

}
jcli-ls(){ jar tvf $(jcli-jar) ; }

jcli-demo(){
   cd $(env-home)/java/commons/cli
   javac -cp $(jcli-jar) Config.java && java -cp .:$(jcli-jar) Config $*
}

