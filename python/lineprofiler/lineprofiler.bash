# === func-gen- : python/lineprofiler/lineprofiler fgp python/lineprofiler/lineprofiler.bash fgn lineprofiler fgh python/lineprofiler
lineprofiler-src(){      echo python/lineprofiler/lineprofiler.bash ; }
lineprofiler-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lineprofiler-src)} ; }
lineprofiler-vi(){       vi $(lineprofiler-source) ; }
lineprofiler-env(){      elocal- ; }
lineprofiler-usage(){ cat << EOU

Python Line Profiler
=====================

https://github.com/rkern/line_profiler

http://www.huyng.com/posts/python-performance-analysis/



EOU
}
lineprofiler-dir(){ echo $(local-base)/env/python/line_profiler-master ; }
lineprofiler-cd(){  cd $(lineprofiler-dir); }
lineprofiler-mate(){ mate $(lineprofiler-dir) ; }
lineprofiler-get(){
   local dir=$(dirname $(lineprofiler-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -f lineprofiler.zip ] && curl -L https://github.com/rkern/line_profiler/archive/master.zip -o lineprofiler.zip
   [ ! -d line_profiler-master ] && unzip lineprofiler.zip


}
