# === func-gen- : java/commons/exec/jexec fgp java/commons/exec/jexec.bash fgn jexec fgh java/commons/exec
jexec-src(){      echo java/commons/exec/jexec.bash ; }
jexec-source(){   echo ${BASH_SOURCE:-$(env-home)/$(jexec-src)} ; }
jexec-vi(){       vi $(jexec-source) ; }
jexec-env(){      
   elocal- ; 
   export APACHE_COMMONS_EXEC_JAR=$(jexec-jar)
}
jexec-usage(){ cat << EOU

JAVA COMMONS EXEC
===================

* http://commons.apache.org/proper/commons-exec/
* http://stackoverflow.com/questions/7340452/process-output-from-apache-commons-exec


EOU
}
jexec-dir(){ echo $(local-base)/env/java/commons/exec/$(jexec-nam) ; }
jexec-cd(){  cd $(jexec-dir); }
jexec-mate(){ mate $(jexec-dir) ; }
jexec-jar(){ echo $(jexec-dir)/$(jexec-nam).jar ; }
jexec-ls(){  jar tvf $(jexec-jar) ; }
#jexec-name(){ echo commons-exec-1.1-src ; }
jexec-name(){ echo commons-exec-1.1-bin ; }

jexec-nam-(){
   case $1 in 
      *bin) echo ${1/-bin} ;;
      *src) echo $1 ;;
   esac
}
jexec-nam(){
   $FUNCNAME- $(jexec-name)
}
jexec-url(){
   case $1 in 
     commons-exec-1.1-src) echo http://ftp.twaren.net/Unix/Web/apache//commons/exec/source/$1.tar.gz ;;
     commons-exec-1.1-bin) echo http://ftp.tc.edu.tw/pub/Apache//commons/exec/binaries/$1.tar.gz  ;;
   esac
}

jexec-get(){
   local dir=$(dirname $(jexec-dir)) &&  mkdir -p $dir && cd $dir

   local name=$(jexec-name)
   local nam=$(jexec-nam $name)
   local url=$(jexec-url $name)
   local tgz=$(basename $url)

   echo $msg name $name nam $nam url $url tgz $tgz

   [ ! -f "$tgz" ] && curl -L -O "$url"
   [ ! -d "$nam" ] && tar zxvf $tgz
}
