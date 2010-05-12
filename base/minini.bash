# === func-gen- : base/minini fgp base/minini.bash fgn minini fgh base
minini-src(){      echo base/minini.bash ; }
minini-source(){   echo ${BASH_SOURCE:-$(env-home)/$(minini-src)} ; }
minini-srcdir(){   echo $(dirname $(minini-source)) ; }
minini-vi(){       vi $(minini-source) ; }
minini-env(){      elocal- ; }
minini-usage(){
  cat << EOU
     minini-src : $(minini-src)
     minini-dir : $(minini-dir)

     http://www.compuphase.com/minini.htm
     http://code.google.com/p/minini

     Contributed the patch getting the C++ class working in ticket :
         http://code.google.com/p/minini/issues/detail?id=11

     No interpolation  ... but probably the need for this 
     can be avoided : no point constructing a silly URL and then
     deconstructing it at the C++ end

EOU
}

minini-rev(){   echo 28 ; }
minini-url(){   echo http://minini.googlecode.com/svn/trunk ; }
minini-name(){  echo minini ; }
minini-dir(){   echo $(local-base)/env/base/$(minini-name) ; }
minini-cd(){  cd $(minini-dir)/$1 ; }
minini-mate(){ mate $(minini-dir) ; }
minini-get(){
   local dir=$(dirname $(minini-dir)) &&  mkdir -p $dir && cd $dir
   svn co $(minini-url)@$(minini-rev) $(minini-name)
   minini-patch
}

minini-patchpath(){ echo $(minini-srcdir)/minini-r$(minini-rev).patch ; }

minini-patch(){
  minini-cd
  local path=$(minini-patchpath)
  [ ! -f "$path" ] && echo $msg no patch $path  && return 0
  patch -p0 < $path   
}


minini-build(){
   minini-cd dev
   gcc -o minIni.o -c minIni.c
   gcc -o test.o   -c test.c
   gcc -o test test.o minIni.o
   ./test
}

minini-test2(){
   minini-cd dev

   gcc -o minIni.o -c minIni.c
   g++ -o test2.o -c test2.cc
   g++ -o test2 test2.o minIni.o
   ./test2
}


