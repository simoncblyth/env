# === func-gen- : doc/doxygen fgp doc/doxygen.bash fgn doxygen fgh doc
doxygen-src(){      echo doc/doxygen.bash ; }
doxygen-source(){   echo ${BASH_SOURCE:-$(env-home)/$(doxygen-src)} ; }
doxygen-vi(){       vi $(doxygen-source) ; }
doxygen-env(){      elocal- ; }
doxygen-usage(){
  cat << EOU
     doxygen-src : $(doxygen-src)
     doxygen-dir : $(doxygen-dir)

     http://www.stack.nl/~dimitri/doxygen/index.html
     http://www.stack.nl/~dimitri/doxygen/manual.html
     http://www.stack.nl/~dimitri/doxygen/starting.html#extract_all

   versions 
       G : 1.5.8 
       N : 1.4.7
       C : 1.3.9.1

   on G fail to install doxygen for want of graphviz ... which needs newer Xcode
{{{
simon:e blyth$ sudo port install doxygen
--->  Installing gmp @4.3.1_0
--->  Installing coreutils @7.4_0
--->  Extracting graphviz
Error: On Mac OS X 10.5, graphviz 2.22.2 requires Xcode 3.1.2 or later but you have Xcode 3.0.
Error: Target org.macports.extract returned: incompatible Xcode version
Error: The following dependencies failed to build: graphviz
Error: Status 1 encountered during processing.
}}}

     Xcode 3.2 requires an Intel-based Mac running Mac OS X Snow Leopard version 10.6.2 or later
     Xcode 3.1.4  ... the last for PPC and 10.5 ?


     


EOU
}
doxygen-name(){ echo doxygen-1.6.3.src ; }
doxygen-url(){ echo http://ftp.stack.nl/pub/users/dimitri/$(doxygen-name).tar.gz ; }
doxygen-dir(){ echo $(local-base)/env/doc/doc-doxygen ; }
doxygen-cd(){  cd $(doxygen-dir); }
doxygen-mate(){ mate $(doxygen-dir) ; }
doxygen-get(){
   local dir=$(dirname $(doxygen-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -f "$(doxygen-name).tar.gz" ] && curl -O $(doxygen-url)
   [ ! -d "$(doxygen-name)"  ]       && tar zxvf $(doxygen-name).tar.gz

}
