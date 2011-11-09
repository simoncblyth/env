cpprun-src(){    echo cpprun/cpprun.bash ; }
cpprun-source(){ echo $(env-home)/$(cpprun-src) ; }
cpprun-dir(){ echo $(dirname $(cpprun-source)); }
cpprun-vi(){     vi $(cpprun-source) ; }
cpprun-cd(){     cd $(cpprun-dir) ; }
cpprun-env(){ elocal- ; }
cpprun-usage(){ cat << EOU

1) create standalone C++ source file <name>.cc (eg seed.cc) 
   containing a main in $(env-home)/cpprun 
2) create bash function named after the source file in cpprun.bash eg "seed"
3) compile and run the main with the <name> bash function, ie "seed"

EOU
}

cpprun(){
   local msg="=== $FUNCNAME :"
   local name=${1:-noname}
   shift 

   local iwd=$(pwd)
   cd $(cpprun-dir)

   local arc=${CMTBIN:-$(uname)}
   local exe=/tmp/$USER/env/$FUNCNAME/$arc/$name
   [ ! -d $(dirname $exe) ] && mkdir -p $(dirname $exe)

   local src=$name.cc
   [ $src -nt $exe ] && echo recompiling $src to create $exe  &&  g++ -o $exe $src 
   $exe $*

   cd $iwd
}

seed(){ cpprun $FUNCNAME ; }
