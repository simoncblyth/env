# === func-gen- : opencl/pyopencl fgp opencl/pyopencl.bash fgn pyopencl fgh opencl
pyopencl-src(){      echo opencl/pyopencl.bash ; }
pyopencl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pyopencl-src)} ; }
pyopencl-vi(){       vi $(pyopencl-source) ; }
pyopencl-env(){      elocal- ; }
pyopencl-usage(){ cat << EOU

PYOPENCL
=========

* http://documen.tician.de/pyopencl/index.html


Installation
-------------

* http://wiki.tiker.net/PyOpenCL/Installation
* http://wiki.tiker.net/PyOpenCL/Installation/Mac



EOU
}
pyopencl-dir(){ echo $(local-base)/env/opencl/pyopencl ; }
pyopencl-cd(){  cd $(pyopencl-dir); }
pyopencl-mate(){ mate $(pyopencl-dir) ; }
pyopencl-get(){
   local dir=$(dirname $(pyopencl-dir)) &&  mkdir -p $dir && cd $dir


   [ ! -d pyopencl ] && git clone http://git.tiker.net/trees/pyopencl.git 

}
