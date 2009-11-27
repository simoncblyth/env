# === func-gen- : pcre/pcre fgp pcre/pcre.bash fgn pcre fgh pcre
pcre-src(){      echo pcre/pcre.bash ; }
pcre-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pcre-src)} ; }
pcre-vi(){       vi $(pcre-source) ; }
pcre-env(){      
   elocal- ; 
   if [ "$(pcre-mode)" == "source" ]; then
      export PCRE_LIBDIR=$(pcre-libdir)
      export PCRE_INCDIR=$(pcre-incdir)
   fi 
}

pcre-info(){ env | grep PCRE_ ; }
pcre-usage(){
  cat << EOU
     pcre-src : $(pcre-src)
     pcre-dir : $(pcre-dir)

     pcre-mode   : $(pcre-mode)
     pcre-libdir : $(pcre-libdir)
     pcre-incdir : $(pcre-incdir)

     envvars that should only be defined on nodes with system pcre too old
     to be usable 

         PCRE_LIBDIR  : $PCRE_LIBDIR
         PCRE_INCDIR  : $PCRE_INCDIR 

     pcre is available from yum, but it without the pcredemo.c

EOU
}
pcre-name(){ echo pcre-8.00 ; }
pcre-dir(){ echo $(local-base)/env/$(pcre-name) ; }
pcre-cd(){  cd $(pcre-dir); }
pcre-mate(){ mate $(pcre-dir) ; }

pcre-build(){
   pcre-get
   pcre-make
}

pcre-get(){
   local dir=$(dirname $(pcre-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(pcre-name)
   local tgz=$nam.tar.gz
   [ ! -f "$tgz" ] && curl -O ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/$tgz
   [ ! -d "$nam" ] && tar zvxf $tgz

}

pcre-mode(){
  case ${1:-$NODE_TAG} in
     P|G1) echo source ;;
        *) echo system ;;
  esac 
}
pcre-libdir(){ 
   case $(pcre-mode $1) in 
     source) echo $(pcre-dir)/.libs ;; 
     system) echo /usr/lib  ;;
   esac
}
pcre-incdir(){ 
   case $(pcre-mode $1) in 
     source) echo $(pcre-dir) ;; 
     system) echo /usr/include/pcre  ;;
   esac
}   



pcre-make(){
   pcre-cd
   ./configure
   make
}


pcre-demo-(){

   gcc -Wall pcredemo.c -I$(pcre-incdir) -L$(pcre-libdir)  -lpcre

  ## fails to compile due to version mismatch between the yum installed pcre (4.5 or 4.6) and the 
  ## tgz grabbed for the demo ... 4.5/6 are not available
  ## 
  ## pcredemo.c:254: error: `PCRE_NOTEMPTY_ATSTART' undeclared (first use in this function)

  local cmd="LD_LIBRARY_PATH=$(pcre-libdir):$LD_LIBRARY_PATH ./a.out  -g '^local (?P<name>\S+)=(?P<val>\S+)' 'local hello=world' "
  echo running $cmd 
  eval $cmd


## [blyth@cms01 pcre]$ ./a.out  -g '^local (?P<name>\S+)=(?P<val>\S+)' 'local hello=world'
##
##Match succeeded at offset 0
## 0: local hello=world
## 1: hello
## 2: world
##Named substrings
##(1) name: hello
##(2)  val: world

}




pcre-demo(){
   cd $(env-home)/pcre    
   pcre-demo-
}

pcre-grep(){
   cd $(env-home)/pcre
   
   gcc -Wall pcregrep.c -I$(pcre-incdir) -L$(pcre-libdir)  -lpcre

   ## usage example ... the matched lines are show in reverse colors
   cat pcregrep.c | ./a.out '^{.*'

}

pcre-read(){

   cd $(env-home)/pcre
   gcc -Wall read.c -I/usr/include/pcre -L/usr/lib  -lpcre

   ./a.out


}
