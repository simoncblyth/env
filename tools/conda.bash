# === func-gen- : tools/conda fgp tools/conda.bash fgn conda fgh tools
conda-src(){      echo tools/conda.bash ; }
conda-source(){   echo ${BASH_SOURCE:-$(env-home)/$(conda-src)} ; }
conda-vi(){       vi $(conda-source) ; }
conda-env(){      elocal- ; }
conda-usage(){ cat << EOU

Conda 
========

Refs
-----

* http://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/
* https://github.com/conda-forge
* https://conda-forge.org/

Travis Oliphant (NumPy originator) on conda
----------------------------------------------- 

* http://technicaldiscovery.blogspot.tw/2013/12/why-i-promote-conda.html

Conda is an open-source, general, cross-platform package manager.  
One could accurately describe it as a
cross-platform hombrew written in Python.  Anyone can use the tool and
related infrastructure to build and distribute whatever packages they
want.

Linkers and Loaders : http://www.iecc.com/linker/






EOU
}
conda-dir(){ echo $(local-base)/env/tools/tools-conda ; }
conda-cd(){  cd $(conda-dir); }
conda-mate(){ mate $(conda-dir) ; }
conda-get(){
   local dir=$(dirname $(conda-dir)) &&  mkdir -p $dir && cd $dir

}
