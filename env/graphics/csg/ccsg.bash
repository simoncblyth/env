# === func-gen- : env/graphics/csg/ccsg fgp env/graphics/csg/ccsg.bash fgn ccsg fgh env/graphics/csg
ccsg-src(){      echo env/graphics/csg/ccsg.bash ; }
ccsg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ccsg-src)} ; }
ccsg-vi(){       vi $(ccsg-source) ; }
ccsg-env(){      elocal- ; }
ccsg-usage(){ cat << EOU

C CSG : A Composite Solid Geometry Modeler (GPL)
==================================================

* http://c-csg.com
* http://c-csg.com/shared_models/


observations
-------------

* uses C as the modelling language, to build CSG_NODE structure
* does not ray trace
* uses marching tetrahedrons within bbox together with point inside solid
  queries to convert CSG model into boundary triangles for output to STL 


build with ttf enabled fails 
-------------------------------


FreeType2 development package (optional, can be disabled in makefile), 
"sudo apt-get install libfreetype6-dev", tested with version 2.5.1

::

    ./src/ttf.c:24:10: fatal error: 'ft2build.h' file not found
    #include <ft2build.h>

build ttf disabled
-------------------

::

    delta:c-csg blyth$ make
    make: Circular cls <- cls dependency dropped.
    echo -ne '\033c'
    -ne \033c
    gcc -Wall -std=c99 -O3 -funroll-loops -ffast-math -g -pthread -I./src -DNOFREETYPE -lm ./src/ttf.c -c -o ./src/ttf.o
    clang: warning: -lm: 'linker' input unused
    gcc -Wall -std=c99 -O3 -funroll-loops -ffast-math -g -pthread -I./src -DNOFREETYPE -lm ./src/noise.c -c -o ./src/noise.o
    clang: warning: -lm: 'linker' input unused
    gcc -Wall -std=c99 -O3 -funroll-loops -ffast-math -g -pthread -I./src -DNOFREETYPE -lm model.c ./src/c-csg.o ./src/stl.o ./src/bb.o ./src/ttf.o ./src/noise.o -o c-csg
    Undefined symbols for architecture x86_64:
      "_adjoint", referenced from:
          _m_invert in c-csg.o
    ld: symbol(s) not found for architecture x86_64
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    make: *** [c-csg] Error 1


::

    46 inline void adjoint(double *m, double *adj_out)
    47 {
    48     adj_out[ 0] =  minor(m, 1, 2, 3, 1, 2, 3);
    49     adj_out[ 1] = -minor(m, 0, 2, 3, 1, 2, 3);
    ..
    64 }
    ..
    74 
    75 int m_invert(double *inv_out, double *m)
    76 {
    77     adjoint(m, inv_out);
    78     double inv_det = 1.0f / det(m);
    79 
    80     for(int i = 0; i < 16; ++i)
    81         inv_out[i] = inv_out[i] * inv_det;
    82 
    83     return 0;
    84 }


* :google:`clang inline function symbol not found `

* http://stackoverflow.com/questions/12844729/linking-error-for-inline-functions






EOU
}

ccsg-url(){ echo http://c-csg.com/c-csg_v1.2.tar.gz ; }
ccsg-nam(){ echo c-csg ; }
ccsg-dir(){ echo $(local-base)/env/env/graphics/csg/$(ccsg-nam) ; }

ccsg-cd(){  cd $(ccsg-dir); }
ccsg-mate(){ mate $(ccsg-dir) ; }
ccsg-get(){
   local dir=$(dirname $(ccsg-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(ccsg-url)
   local nam=$(ccsg-nam)

   local tgz=$(basename $url)

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz
   


}
