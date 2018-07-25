# === func-gen- : goofit/goofit fgp goofit/goofit.bash fgn goofit fgh goofit
goofit-src(){      echo goofit/goofit.bash ; }
goofit-source(){   echo ${BASH_SOURCE:-$(env-home)/$(goofit-src)} ; }
goofit-vi(){       vi $(goofit-source) ; }
goofit-env(){      elocal- ; }
goofit-usage(){ cat << EOU


* https://goofit.github.io
* https://github.com/GooFit/GooFit
* https://github.com/GooFit
* https://github.com/GooFit/GooTorial



Externals
-----------


Minuit2
~~~~~~~~~

* https://github.com/GooFit/Minuit2
* https://root.cern.ch/root/htmldoc/guides/users-guide/ROOTUsersGuide.html#minuit2-package
* https://github.com/GooFit/Minuit2/blob/master/DEVELOP.md
* http://seal.web.cern.ch/seal/snapshot/work-packages/mathlibs/minuit/doc/doc.html
* http://seal.web.cern.ch/seal/documents/minuit/mntutorial.pdf

Eigen
~~~~~~

* https://github.com/eigenteam/eigen-git-mirror
* https://bitbucket.org/eigen/eigen/src/default/

generics
~~~~~~~~~~

* https://github.com/BryanCatanzaro/generics

NVIDIA GPUs of CUDA compute capability 3.5 and greater, such as the Tesla K20,
support __ldg(), an intrinsic that loads through the read-only texture cache,
and can improve performance in some circumstances. This library allows __ldg to
work on arbitrary types, as detailed below. It also generalizes __shfl() to
shuffle arbitrary types.


modern_cmake
~~~~~~~~~~~~~~

* https://github.com/CLIUtils/modern_cmake

CLI11
~~~~~~~~

* https://github.com/CLIUtils/CLI11

powerful command line parser, with a beautiful, minimal syntax and no
dependencies beyond C++11. It is header only, and comes in a single file form
for easy inclusion in projects. It is easy to use for small projects, but
powerful enough for complex command line projects, and can be customized for
frameworks.



EOU
}
goofit-dir(){ echo $(local-base)/env/goofit/GooFit ; }
goofit-cd(){  cd $(goofit-dir); }
goofit-get(){
   local dir=$(dirname $(goofit-dir)) &&  mkdir -p $dir && cd $dir

   #[ ! -d GooFit ] && git clone git@github.com:simoncblyth/GooFit.git  
   [ ! -d GooFit ] && git clone git://github.com/GooFit/GooFit.git --recursive
   

}
