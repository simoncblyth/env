# === func-gen- : muon_simulation/chroma/chroma fgp muon_simulation/chroma/chroma.bash fgn chroma fgh muon_simulation/chroma
chroma-src(){      echo muon_simulation/chroma/chroma.bash ; }
chroma-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chroma-src)} ; }
chroma-vi(){       vi $(chroma-source) ; }
chroma-usage(){ cat << EOU

CHROMA
=======

* http://chroma.bitbucket.org/install/overview.html
* http://chroma.bitbucket.org/install/macosx.html
* http://chroma.bitbucket.org/install/overview.html#common-install

Chroma Dependencies
--------------------

OSX 10.9.1 Mavericks 
~~~~~~~~~~~~~~~~~~~~~~

* Xcode 5.0.2 with commandline tools
* XQuartz 2.7.5
* CUDA 5.5
* Macports (logs in ~/macports/)

  * py27-matplotlib 
  * mercurial 
  * py27-game 
  * py27-virtualenv 
  * Xft2 
  * xpm

Common
~~~~~~~~

* http://shrinkwrap.readthedocs.org/en/latest/


chroma-deps pycuda build failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://wiki.tiker.net/PyCuda/Installation/Mac

While doing::

   pip install -b /usr/local/env/chroma_env/build/build_pycuda pycuda

#. linker errors from missing dir /Developer/NVIDIA/CUDA-5.5/lib64, avoid with ~/.aksetup-defaults.py change
  
#. linker errors from missing lib "-lcuda", avoid by changing libdir to /usr/local/cuda/lib  


::

   mv /Users/blyth/.pip/pip.log ~/chroma_deps.log


EOU
}
chroma-dir(){ echo $(local-base)/env/chroma_env ; }
chroma-env(){      
    elocal-  
    local dir=$(chroma-dir)
    [ -d $dir ] && source $dir/bin/activate 
}
chroma-cd(){  cd $(chroma-dir); }
chroma-mate(){ mate $(chroma-dir) ; }
chroma-get(){
   local dir=$(dirname $(chroma-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://bitbucket.org/chroma/chroma
}


chroma-prepare(){
    chroma-virtualenv
    chroma-deps
}


chroma-virtualenv(){
    local msg="=== $FUNCNAME :"
    local dir=$(chroma-dir)
    [ -d $dir ] && echo $msg chroma virtualenv dir exists already $dir && return 0

    # want access to macports py27 modules like numpy/pygame/matplotlib/... 
    # so use the somewhat dirty --system-site-package option

    which python
    python -V
    which virtualenv
    virtualenv --version

    virtualenv --system-site-package  $(chroma-dir)    
}

chroma-pycuda-aksetup(){  
   local out=~/.aksetup-defaults.py
   echo $msg writing $out
   $FUNCNAME- > $out
 }
chroma-pycuda-aksetup-(){  
   cuda-
   cat << EOS
# $FUNCNAME
import os
virtual_env = os.environ['VIRTUAL_ENV']
cuda_root = '$(cuda-dir)'
cuda_lib_dir = [os.path.join(cuda_root,'lib')]
cuda_lib_dir = ['/usr/local/cuda/lib']
BOOST_INC_DIR = [os.path.join(virtual_env, 'include')]
BOOST_LIB_DIR = [os.path.join(virtual_env, 'lib')]
BOOST_PYTHON_LIBNAME = ['boost_python']

# guess based on 
#   http://wiki.tiker.net/PyCuda/Installation/Mac
#   https://github.com/inducer/pycuda/blob/master/setup.py
#
CUDADRV_LIB_DIR = cuda_lib_dir
CUDART_LIB_DIR = cuda_lib_dir
CURAND_LIB_DIR = cuda_lib_dir

EOS
}

chroma-deps(){
   chroma-  # activate the virtualenv
   [ -z "$VIRTUAL_ENV" ] && echo $msg ERROR need to be in the virtualenv to proceed && return 1

   cuda-
   chroma-pycuda-aksetup

   export PIP_EXTRA_INDEX_URL=http://mtrr.org/chroma_pkgs/
   which pip

   pip install chroma_deps
}
