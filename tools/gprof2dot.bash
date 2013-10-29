# === func-gen- : tools/gprof2dot fgp tools/gprof2dot.bash fgn gprof2dot fgh tools
gprof2dot-src(){      echo tools/gprof2dot.bash ; }
gprof2dot-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gprof2dot-src)} ; }
gprof2dot-vi(){       vi $(gprof2dot-source) ; }
gprof2dot-env(){      elocal- ; }
gprof2dot-usage(){ cat << EOU

GPROF2DOT
==============

Generate a dot graph from the output of several profilers.

* http://gprof2dot.jrfonseca.googlecode.com/git/gprof2dot.py
* http://code.google.com/p/jrfonseca/w/list
* http://code.google.com/p/jrfonseca/wiki/Gprof2Dot
* http://bioportal.weizmann.ac.il/course/python/PyMOTW/PyMOTW/docs/profile/index.html


FUNCTIONS
---------


Usage
-------

::

    python -m cProfile -o vnode.cprofile $(which vnode.py) --daesave --subcopy -O 000.xml 0 
          # create the vnode.cprofile raw profiling output 

    gprof2dot.py -f pstats vnode.cprofile | dot -Tpng -o vnode.png
    gprof2dot.py -f pstats vnode.cprofile | dot -Tsvg -o vnode.svg
          # cprofile is "pstats" format,  
          # generate a dot graph and thence create PNG or SVG
 
    ln -s $PWD $(apache-htdocs)/collada
    open http://locahost/collada/vnode.svg


EOU
}
gprof2dot-dir(){ echo $(local-base)/env/tools/tools-gprof2dot ; }
gprof2dot-cd(){  cd $(gprof2dot-dir); }
gprof2dot-mate(){ mate $(gprof2dot-dir) ; }
gprof2dot-get(){
   local dir=$(dirname $(gprof2dot-dir)) &&  mkdir -p $dir && cd $dir

   cd $LOCAL_BASE/env/bin && curl -L -O http://gprof2dot.jrfonseca.googlecode.com/git/gprof2dot.py


}

gprof2dot-svg(){
  local prof=${1:-vnode.cprofile}
  local cmd="gprof2dot.py -f pstats $prof | dot -Tsvg -o $prof.svg"
  echo $msg $cmd
  eval $cmd
}


  


